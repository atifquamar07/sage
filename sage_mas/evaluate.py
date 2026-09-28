"""``sage-eval``: grade SAGE answers with xFinder + xVerify (the paper's protocol).

    sage-eval results/qwen2.5-1.5b-instruct/seed_2025/gsm8k.jsonl

For each answer, xFinder extracts the final answer from the response (it never
sees the reference). xVerify-0.5B-I then judges the extraction against the
reference answer. Grades go to ``<dataset>.eval.jsonl`` next to the results;
re-running resumes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
from pathlib import Path

from .answer.regex import extract_ground_truth_answer
from .answer.xfinder import XFinderClient
from .config import load_model_config
from .data import load_dataset, read_jsonl
from .llm.client import ChatRequest, OpenAICompatClient
from .third_party.xverify_prompt import format_prompt

REPO = Path(__file__).resolve().parents[1]
LABEL_RE = re.compile(r"^\s*\[?(correct|incorrect)\]?\s*$", re.IGNORECASE)


def reference_answer(row: dict) -> str:
    for field in ("gt_answer", "answer", "gold_answer"):
        if row.get(field) is not None and str(row[field]).strip():
            return str(row[field]).strip()
    return extract_ground_truth_answer(str(row.get("gt", ""))).strip()


class Grader:
    def __init__(self, xfinder: XFinderClient, judge: OpenAICompatClient, judge_model: str):
        self.xfinder, self.judge, self.judge_model = xfinder, judge, judge_model

    def grade(self, query: str, response: str, choices: list, reference: str) -> dict:
        if not reference:
            raise ValueError("no reference answer")
        extracted = self.xfinder.extract(query, response, choices).answer
        prompt = format_prompt(query, extracted or "[No valid answer]", reference)
        completion = self.judge.complete(
            ChatRequest(model=self.judge_model, system=None, prompt=prompt, temperature=0.0, stage="evaluation")
        )
        if completion.finish_reason not in (None, "stop"):
            raise ValueError(f"xVerify stopped with {completion.finish_reason!r}")
        match = LABEL_RE.match(completion.text or "")
        if match is None:
            raise ValueError(f"xVerify returned an invalid label: {completion.text!r}")
        return {
            "correct": match.group(1).lower() == "correct",
            "extracted_answer": extracted,
            "reference_answer": reference,
            "judge_output": completion.text,
        }


def evaluate_file(results_path: Path, grader: Grader, workers: int, attempts: int = 3) -> tuple[int, int, int]:
    """Grade one results file; returns ``(graded, correct, failed)``."""
    manifest_path = results_path.with_name(results_path.stem + ".manifest.json")
    dataset = json.loads(manifest_path.read_text())["dataset"] if manifest_path.exists() else results_path.stem
    gold = {row["sample_id"]: row for row in load_dataset(dataset)}
    eval_path = results_path.with_name(results_path.stem + ".eval.jsonl")
    done = {}
    if eval_path.exists():
        for line in read_jsonl(eval_path):
            try:
                record = json.loads(line)
                done[record["sample_id"]] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    rows = [json.loads(line) for line in read_jsonl(results_path)]
    todo = [row for row in rows if row["sample_id"] not in done]
    lock, failed = threading.Lock(), []

    def work(row: dict) -> None:
        reference = reference_answer(gold[row["sample_id"]])
        error: str | None = None
        for _ in range(attempts):
            try:
                grade = grader.grade(row["query"], row["answer"], row.get("choices") or [], reference)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            with lock:
                failed.append({"sample_id": row["sample_id"], "error": error})
            return
        record = {"sample_id": row["sample_id"], **grade}
        with lock:
            done[row["sample_id"]] = record
            with eval_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    graded = [done[row["sample_id"]] for row in rows if row["sample_id"] in done]
    for failure in failed:
        print(f"[sage-eval] {results_path.name} {failure['sample_id']}: {failure['error']}", file=sys.stderr)
    return len(graded), sum(bool(g["correct"]) for g in graded), len(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sage-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="results .jsonl files written by sage-run")
    parser.add_argument("--xfinder", type=Path, default=REPO / "configs" / "models" / "xfinder-qwen1505.yaml")
    parser.add_argument("--judge", type=Path, default=REPO / "configs" / "models" / "xverify-0.5b-i.yaml")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args(argv)

    xf, judge = load_model_config(args.xfinder), load_model_config(args.judge)
    grader = Grader(
        XFinderClient(xf.endpoints, xf.name, max_tokens=xf.max_tokens, timeout=xf.timeout),
        OpenAICompatClient({judge.name: judge}),
        judge.name,
    )
    status = 0
    # Accept globs like results/*/seed_*/*.jsonl: skip grade and error files.
    paths = [p for p in args.results if not p.name.endswith((".eval.jsonl", ".errors.jsonl"))]
    for path in paths:
        graded, correct, failed = evaluate_file(path, grader, args.workers)
        accuracy = 100.0 * correct / graded if graded else float("nan")
        print(f"[sage-eval] {path}: {correct}/{graded} correct ({accuracy:.2f}%), {failed} failed")
        status |= int(failed > 0)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
