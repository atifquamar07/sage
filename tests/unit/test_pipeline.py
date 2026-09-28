"""Configuration, data files and aggregation."""

import json

import pytest

from sage_mas.aggregate import collect, render, seed_stats
from sage_mas.config import SageConfig, load_sage_config
from sage_mas.data import DATASETS, INFERENCE_FIELDS, inference_view, load_dataset
from sage_mas.evaluate import reference_answer

REPO = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_default_config_is_the_paper_setting():
    config = SageConfig()
    assert (config.num_agents, config.routing.top_k, config.routing.max_rounds) == (4, 2, 3)
    assert (config.prefix.tau, config.prefix.weight, config.review.sample_size, config.vote.leader_bonus) == (
        0.6,
        0.5,
        2,
        0.5,
    )
    assert load_sage_config(REPO / "configs" / "sage.yaml") == config


def test_config_overrides_and_validation():
    config = load_sage_config(None, ["num_agents=9", "routing.top_k=3", "answer_extractor=regex"])
    assert (config.num_agents, config.routing.top_k, config.answer_extractor) == (9, 3, "regex")
    with pytest.raises(ValueError):
        load_sage_config(None, ["routing.nope=1"])
    with pytest.raises(ValueError):
        load_sage_config(None, ["routing.top_k=4"])  # K must be < N


@pytest.mark.parametrize("name", [d for d in DATASETS if d != "gpqa_diamond"])
def test_shipped_datasets(name):
    rows = load_dataset(name)  # also verifies the checksum
    assert len(rows) == 500
    view = inference_view(rows[0])
    assert set(view) <= set(INFERENCE_FIELDS) and "gt" not in view and "gt_answer" not in view
    assert reference_answer(rows[0])


def test_gpqa_text_is_not_shipped():
    manifest = json.loads((REPO / "data" / "gpqa_diamond.manifest.json").read_text())
    assert len(manifest["samples"]) == 198
    assert all(set(s) == {"sample_id", "source_row_index", "row_hash"} for s in manifest["samples"])


def test_seed_stats_mean_sd_and_interval():
    stats = seed_stats([60.0, 70.0, 80.0])
    assert stats["mean"] == 70.0 and stats["sd"] == 10.0
    assert [round(x, 2) for x in stats["ci95"]] == [45.16, 94.84]


def test_aggregate_reads_eval_files(tmp_path):
    for seed, correct in [(1, 3), (2, 2)]:
        folder = tmp_path / f"seed_{seed}"
        folder.mkdir()
        (folder / "gsm8k.jsonl").write_text("".join(json.dumps({"sample_id": i}) + "\n" for i in range(4)))
        (folder / "gsm8k.eval.jsonl").write_text(
            "".join(json.dumps({"sample_id": i, "correct": i < correct}) + "\n" for i in range(4))
        )
    table = collect(tmp_path)
    assert table == {"gsm8k": {1: 75.0, 2: 50.0}}
    assert "62.50 ± 17.68" in render(table)


def test_read_jsonl_keeps_unicode_line_separators(tmp_path):
    from sage_mas.data import read_jsonl

    path = tmp_path / "rows.jsonl"
    rows = [{"text": "a b\x85c"}, {"text": "d"}]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert [json.loads(line) for line in read_jsonl(path)] == rows
