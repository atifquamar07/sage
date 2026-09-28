#!/usr/bin/env python3
"""Download GPQA-Diamond and rebuild ``data/gpqa_diamond.json``.

GPQA's authors ask that its questions not be redistributed in plain text, so the
repository ships only ``data/gpqa_diamond.manifest.json`` (sample ids and row
hashes). This script downloads the pinned Hugging Face revision used in the
paper, rebuilds the 198 evaluation rows, and checks every row hash.

    pip install "sage-mas[data]"   # provides the `datasets` package
    python scripts/fetch_gpqa.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO_ID = "fingertap/GPQA-Diamond"
REVISION = "68be7564497676e07a77a042fdb587deb88c51c3"
SPLIT = "test"
LETTERS = ("A", "B", "C", "D")
FINAL_CHOICE_INSTRUCTION = (
    "\n\nSolve step by step. Put only the final option letter on the last line "
    "in the form \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
)
_OPTION_MARKER_RE = re.compile(r"(?m)^[ \t]*([A-D])[.)][ \t]+")
DATA = Path(__file__).resolve().parents[1] / "data"


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def split_choices(question: str) -> list[str]:
    """The last complete, line-initial ``A.``-``D.`` option block of a question."""
    markers = list(_OPTION_MARKER_RE.finditer(question))
    block = None
    for start in range(len(markers) - 3):
        if tuple(m.group(1) for m in markers[start : start + 4]) == LETTERS:
            block = markers[start : start + 4]
    if block is None:
        raise ValueError("question has no complete A-D option block")
    return [question[m.end() : (block[i + 1].start() if i < 3 else len(question))].strip() for i, m in enumerate(block)]


def build_rows(raw_rows, manifest: dict) -> list[dict]:
    rows = []
    for index, raw in enumerate(raw_rows):
        question = str(raw["question"])
        choices = split_choices(question)
        answer = str(raw["answer"]).strip().upper()
        row = {
            "query": question.strip() + FINAL_CHOICE_INSTRUCTION,
            "gt": f"({answer}) {choices[LETTERS.index(answer)]}",
            "gt_answer": answer,
            "choices": choices,
            "answer_type": "choice",
            "tag": ["science", "GPQA", "GPQA-Diamond"],
            "source": "GPQA-Diamond",
        }
        identity = {
            "dataset": manifest["dataset_name"],
            "source_dataset": manifest["source_dataset"],
            "source_split": manifest["source_split"],
            "source_row_index": index,
            "query": row["query"],
            "gt": row["gt"],
            "gt_answer": row["gt_answer"],
            "choices": row["choices"],
        }
        row["sample_id"] = stable_hash(identity)[:24]
        row["source_row_index"] = index
        row["dataset_adapter_version"] = manifest["dataset_adapter_version"]
        row["dataset_name"] = manifest["dataset_name"]
        row["dataset_row_hash"] = stable_hash(
            {
                key: row.get(key)
                for key in ("query", "gt", "gt_answer", "answer", "choices", "answer_type", "numeric_tolerance")
            }
        )
        row["dataset_manifest_hash"] = manifest["dataset_manifest_hash"]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA / "gpqa_diamond.json")
    args = parser.parse_args()

    from datasets import load_dataset

    manifest = json.loads((DATA / "gpqa_diamond.manifest.json").read_text())
    body = {k: v for k, v in manifest.items() if k != "dataset_manifest_hash"}
    if stable_hash(body) != manifest["dataset_manifest_hash"]:
        raise SystemExit("gpqa_diamond.manifest.json is corrupted")

    rows = build_rows(load_dataset(REPO_ID, split=SPLIT, revision=REVISION), manifest)
    expected = manifest["samples"]
    if len(rows) != len(expected):
        raise SystemExit(f"expected {len(expected)} rows, downloaded {len(rows)}")
    for row, entry in zip(rows, expected, strict=True):
        if (row["sample_id"], row["source_row_index"], row["dataset_row_hash"]) != (
            entry["sample_id"],
            entry["source_row_index"],
            entry["row_hash"],
        ):
            raise SystemExit(f"row {row['source_row_index']} does not match the manifest; is the revision pinned?")

    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} verified rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
