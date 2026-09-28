"""Evaluation datasets shipped in ``data/``.

Each file is a JSON list of questions. Besides ``query`` (and ``choices`` for
multiple choice), rows carry frozen identity fields (``sample_id``,
``dataset_name``, ``dataset_manifest_hash``, ``dataset_row_hash``,
``source_row_index``). They seed the per-question role sampling, so they are kept
exactly as in the paper runs. Reference answers are removed before a question is
passed to SAGE.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DATASETS = ("gsm8k", "math", "aqua_rat", "gsm_hard", "mmlu", "gpqa_diamond")

# Fields SAGE may see. Everything else (reference answers, sources, metadata) is withheld.
INFERENCE_FIELDS = (
    "query",
    "choices",
    "sample_id",
    "dataset_name",
    "dataset_manifest_hash",
    "dataset_row_hash",
    "source_row_index",
)


def data_dir() -> Path:
    return Path(os.environ.get("SAGE_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")


def dataset_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".json" and path.exists():
        return path
    if name_or_path not in DATASETS:
        raise ValueError(f"unknown dataset {name_or_path!r}; choose from {', '.join(DATASETS)} or pass a .json path")
    path = data_dir() / f"{name_or_path}.json"
    if not path.exists():
        hint = " (run scripts/fetch_gpqa.py first)" if name_or_path == "gpqa_diamond" else ""
        raise FileNotFoundError(f"{path} not found{hint}")
    return path


def load_dataset(name_or_path: str, *, verify: bool = True) -> list[dict[str, Any]]:
    path = dataset_path(name_or_path)
    if verify:
        verify_checksum(path)
    rows = json.loads(path.read_text())
    ids = [row.get("sample_id") for row in rows]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError(f"{path}: every row needs a unique sample_id")
    return rows


def verify_checksum(path: Path) -> None:
    """Check a shipped dataset file against ``data/manifest.json`` (other files are skipped)."""
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        return
    entries = json.loads(manifest_path.read_text()).get("datasets", {})
    expected: str | None = next((e.get("sha256") for e in entries.values() if e.get("file") == path.name), None)
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{path} does not match the checksum in {manifest_path}")


def read_jsonl(path: Path) -> list[str]:
    """Non-empty lines of a JSON Lines file.

    Splits on newlines only; ``str.splitlines`` would also split on characters such
    as U+2028 that may appear inside JSON strings.
    """
    return [line for line in Path(path).read_text().split("\n") if line.strip()]


def inference_view(row: dict[str, Any]) -> dict[str, Any]:
    """The question as SAGE sees it: no reference answer or other metadata."""
    return {field: row[field] for field in INFERENCE_FIELDS if field in row}
