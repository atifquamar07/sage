"""Reciprocal peer review and donor selection (Eq. 4-5)."""

import random

import pytest
from conftest import make_context

from sage_mas.core.review import parse_critique, peer_review, sample_group, split_groups


def test_groups_top_score_plus_next_best():
    high, low, _ = split_groups([1.0, 1.5, 0.5, 1.0], random.Random(0))
    assert high == [1, 0] and low == [2, 3]  # next best: highest score, then lowest index


def test_groups_all_top_ties_go_high():
    high, low, _ = split_groups([1.5, 1.5, 1.5, 0.5], random.Random(0))
    assert high == [0, 1, 2] and low == [3]


def test_groups_all_equal_draw_two_at_random():
    draws = {tuple(split_groups([1.0] * 4, random.Random(seed))[0]) for seed in range(30)}
    assert all(len(d) == 2 for d in draws) and len(draws) > 1


def test_sample_group():
    rng = random.Random(0)
    assert sample_group([3, 1], 2, rng) == [3, 1]  # small groups keep their order
    picked = sample_group([0, 1, 2], 2, rng)
    assert len(picked) == 2 and picked == sorted(picked)
    assert sample_group([], 2, rng) == []


def test_parse_critique():
    assert parse_critique("action: EDIT\nfinal_answer:\nsteps\nFinal answer: \\boxed{3}") == (
        "EDIT",
        "steps\nFinal answer: \\boxed{3}",
    )
    assert parse_critique("Action: `keep`\nfinal_answer: x")[0] == "KEEP"
    for bad in [
        "**Action:** EDIT\nfinal_answer: x",  # markdown labels are not accepted
        "action: KEEP\naction: EDIT\nfinal_answer: x",
        "final_answer: x\naction: KEEP",
        "action: KEEP\nfinal_answer:\n```",
        "no envelope",
    ]:
        with pytest.raises(ValueError):
            parse_critique(bad)


def test_peer_review_selects_donor_and_falls_back_on_bad_reviews():
    answers = ["\\boxed{4}", "\\boxed{4}", "\\boxed{7}", "\\boxed{8}"]

    def respond(request):
        if request.stage == "reciprocal_comparison":
            if request.agent == 3:
                return "garbled"  # parse failure -> KEEP own answer
            return "action: EDIT\nfinal_answer:\nFinal answer: \\boxed{4}"
        return "\\boxed{4}"  # prefix completions

    ctx = make_context(respond)
    result = peer_review(ctx, answers, [1.5, 1.5, 0.75, 0.75])
    assert result.high == [0, 1] and result.low == [2, 3]
    assert len(result.reviews) == 8  # 2 x 2 pairs, both directions
    assert [r.owner for r in result.reviews[:2]] == [0, 2]
    assert all(r.error for r in result.reviews if r.owner == 3)
    assert {k.agent for k in result.retained} == {0, 1, 2, 3}
    assert result.donor == 0  # ties on the retained score break by ρ⁰, then lower index
