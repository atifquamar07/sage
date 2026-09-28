"""``sage-aggregate``: accuracy per dataset over seeds (mean ± sample SD, 95% CI).

    sage-aggregate results/qwen2.5-1.5b-instruct

Reads every ``seed_*/<dataset>.eval.jsonl`` below the given folder. AVG is the
macro-average: the benchmark accuracies are averaged within each seed first.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .data import DATASETS, read_jsonl

# Two-sided 95% Student-t critical values by degrees of freedom.
_T975 = {
    1: 12.706204736174698,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818366147395,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.306004135033371,
    9: 2.262157162740992,
    10: 2.2281388519649385,
    15: 2.131449545559323,
    20: 2.0859634472658364,
    30: 2.0422724563012373,
}


def t_critical(df: int) -> float:
    if df in _T975:
        return _T975[df]
    below = [d for d in _T975 if d < df]
    return _T975[max(below)] if df <= 30 else 1.959963984540054


def seed_stats(values: Sequence[float]) -> dict:
    """Mean, sample SD and two-sided t 95% CI of per-seed accuracies (in %)."""
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = t_critical(len(values) - 1) * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sd": sd, "ci95": (mean - margin, mean + margin)}


def collect(folder: Path) -> dict[str, dict[int, float]]:
    """``{dataset: {seed: accuracy %}}`` for every eval file below ``folder``."""
    table: dict[str, dict[int, float]] = defaultdict(dict)
    for path in sorted(folder.glob("seed_*/*.eval.jsonl")):
        seed = int(path.parent.name.removeprefix("seed_"))
        dataset = path.name.removesuffix(".eval.jsonl")
        grades = [json.loads(line) for line in read_jsonl(path)]
        results = path.with_name(f"{dataset}.jsonl")
        expected = len(read_jsonl(results)) if results.exists() else len(grades)
        if len(grades) != expected:
            raise ValueError(f"{path}: {len(grades)} grades for {expected} results; finish sage-eval first")
        table[dataset][seed] = 100.0 * sum(bool(g["correct"]) for g in grades) / len(grades)
    return dict(table)


def render(table: dict[str, dict[int, float]]) -> str:
    datasets = [d for d in DATASETS if d in table] + sorted(d for d in table if d not in DATASETS)
    seeds = sorted({seed for per_seed in table.values() for seed in per_seed})
    lines = [
        "| Dataset | " + " | ".join(f"seed {s}" for s in seeds) + " | Mean ± SD | 95% CI |",
        "|---|" + "---|" * (len(seeds) + 2),
    ]
    for dataset in datasets:
        values = [table[dataset][s] for s in seeds if s in table[dataset]]
        stats = seed_stats(values)
        cells = [f"{table[dataset][s]:.2f}" if s in table[dataset] else "–" for s in seeds]
        lines.append(
            f"| {dataset} | "
            + " | ".join(cells)
            + f" | {stats['mean']:.2f} ± {stats['sd']:.2f} | [{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}] |"
        )
    complete = [s for s in seeds if all(s in table[d] for d in datasets)]
    if len(datasets) > 1 and complete:
        averages = {s: statistics.mean(table[d][s] for d in datasets) for s in complete}
        stats = seed_stats(list(averages.values()))
        cells = [f"{averages[s]:.2f}" if s in averages else "–" for s in seeds]
        lines.append(
            "| **AVG** | "
            + " | ".join(cells)
            + f" | {stats['mean']:.2f} ± {stats['sd']:.2f} | [{stats['ci95'][0]:.2f}, {stats['ci95'][1]:.2f}] |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sage-aggregate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path, help="results/<model> folder containing seed_* subfolders")
    parser.add_argument("--json", action="store_true", help="print raw per-seed accuracies as JSON")
    args = parser.parse_args(argv)
    table = collect(args.folder)
    if not table:
        parser.error(f"no seed_*/*.eval.jsonl files under {args.folder}")
    print(json.dumps(table, indent=2) if args.json else render(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
