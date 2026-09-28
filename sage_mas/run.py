"""``sage-run``: run SAGE on a dataset and write one JSON line per question.

    sage-run --model configs/models/qwen2.5-1.5b-instruct.yaml --dataset gsm8k --seed 2025

Results go to ``<out>/<model>/seed_<seed>/<dataset>.jsonl``. Re-running the same
command resumes: finished questions are skipped, and questions that failed are
retried. Questions that fail again are listed in ``<dataset>.errors.jsonl``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import __version__
from .answer.extractor import RegexExtractor, XFinderExtractor
from .answer.xfinder import XFinderClient
from .config import ModelConfig, SageConfig, config_dict, load_model_config, load_sage_config
from .data import DATASETS, dataset_path, inference_view, load_dataset, read_jsonl
from .llm.client import OpenAICompatClient
from .sage import SAGE
from .seeding import sha256_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT_XFINDER = REPO / "configs" / "models" / "xfinder-qwen1505.yaml"


def build_sage(config: SageConfig, models: list[ModelConfig], xfinder: ModelConfig | None) -> SAGE:
    names = [m.name for m in models]
    missing = sorted(set(config.backbones or []) - set(names))
    if missing:
        raise ValueError(f"backbones {missing} have no --model config")
    if config.answer_extractor == "xfinder":
        if xfinder is None:
            raise ValueError("answer_extractor=xfinder needs --xfinder (or set answer_extractor=regex)")
        extractor = XFinderExtractor(
            XFinderClient(xfinder.endpoints, xfinder.name, max_tokens=xfinder.max_tokens, timeout=xfinder.timeout)
        )
    else:
        extractor = RegexExtractor()
    llm = OpenAICompatClient({m.name: m for m in models})
    return SAGE(config, llm, extractor, model=names[0])


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest(
    config: SageConfig, models: list[ModelConfig], xfinder: ModelConfig | None, dataset: str, seed: int
) -> dict:
    def model_entry(m: ModelConfig) -> dict:
        entry = config_dict(m)
        entry.pop("endpoints")  # where a server ran does not affect results
        return entry

    settings = {
        "config": config_dict(config),
        "models": [model_entry(m) for m in models],
        "xfinder": model_entry(xfinder) if xfinder and config.answer_extractor == "xfinder" else None,
        "dataset": dataset,
        "seed": seed,
    }
    return {
        **settings,
        "settings_sha256": sha256_json(settings),
        "sage_mas_version": __version__,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _read_done(path: Path) -> dict[str, str]:
    """Completed rows by sample id; drops a torn final line left by an interrupted run."""
    done, lines = {}, []
    if path.exists():
        for line in read_jsonl(path):
            try:
                done[json.loads(line)["sample_id"]] = line
                lines.append(line)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        path.write_text("".join(f"{line}\n" for line in lines))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sage-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=Path,
        help="model config; repeat for per-agent backbones (see config 'backbones')",
    )
    parser.add_argument("--config", type=Path, help="SAGE config (default: paper settings)")
    parser.add_argument("--xfinder", type=Path, default=DEFAULT_XFINDER, help="xFinder model config")
    parser.add_argument("--dataset", required=True, help="dataset name (e.g. gsm8k) or path to a .json file")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--workers", type=int, default=64, help="questions processed concurrently")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--name", help="result folder name (default: the model name)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config field")
    parser.add_argument("--trace-prompts", action="store_true", help="store full prompts in the call records")
    args = parser.parse_args(argv)

    config = load_sage_config(args.config, args.set)
    models = [load_model_config(path) for path in args.model]
    xfinder = load_model_config(args.xfinder) if config.answer_extractor == "xfinder" else None
    sage = build_sage(config, models, xfinder)

    path = dataset_path(args.dataset)
    dataset = path.stem
    rows = load_dataset(args.dataset)[: args.limit]
    name = args.name or ("mixed" if config.backbones and len(set(config.backbones)) > 1 else models[0].name)
    out_dir = args.out / name / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path, manifest_path = out_dir / f"{dataset}.jsonl", out_dir / f"{dataset}.manifest.json"
    errors_path = out_dir / f"{dataset}.errors.jsonl"

    # Shipped datasets are recorded by name, other files by path (sage-eval reads their references).
    source = args.dataset if args.dataset in DATASETS else str(path.resolve())
    manifest = _manifest(config, models, xfinder, source, args.seed)
    if manifest_path.exists() and out_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("settings_sha256") != manifest["settings_sha256"]:
            parser.error(f"{out_path} was produced with different settings; use another --out or --name")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    done = _read_done(out_path)
    todo = [row for row in rows if row["sample_id"] not in done]
    print(f"[sage-run] {dataset} seed {args.seed}: {len(rows) - len(todo)} done, {len(todo)} to run -> {out_path}")

    lock = threading.Lock()
    failures: list[dict] = []

    def work(row: dict) -> None:
        try:
            result = sage.run(inference_view(row), args.seed)
        except Exception as exc:  # recorded and retried on the next invocation
            with lock:
                failures.append({"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}"})
            return
        record = {
            "sample_id": row["sample_id"],
            "dataset": dataset,
            "seed": args.seed,
            "query": row["query"],
            **({"choices": row["choices"]} if row.get("choices") else {}),
            **result.to_dict(full_prompts=args.trace_prompts),
        }
        line = json.dumps(record, ensure_ascii=False)
        with lock, out_path.open("a") as handle:
            handle.write(line + "\n")

    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for _ in concurrent.futures.as_completed([pool.submit(work, row) for row in todo]):
            finished += 1
            if finished % 50 == 0 or finished == len(todo):
                print(f"[sage-run] {finished}/{len(todo)} ({len(failures)} failed)", flush=True)

    errors_path.write_text("".join(json.dumps(f) + "\n" for f in failures))
    if failures:
        print(f"[sage-run] {len(failures)} questions failed; see {errors_path}. Re-run to retry them.", file=sys.stderr)
        return 1
    errors_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
