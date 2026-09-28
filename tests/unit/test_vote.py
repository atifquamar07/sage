"""Weighted pool vote (Eq. 9) and its tie-breaks."""

from sage_mas.core.vote import INITIAL, REVIEW, ROUND, Candidate, best_index, pool_vote, priority


def pool(*entries):
    return [
        Candidate(src, rnd, agent, f"{key}-{i}", key, leader, i)
        for i, (src, rnd, agent, key, leader) in enumerate(entries)
    ]


def test_leader_weight_decides_close_votes():
    candidates = pool(
        (INITIAL, 0, 0, "a", True),
        (INITIAL, 0, 1, "b", False),
        (ROUND, 1, 0, "b", False),
        (ROUND, 1, 1, "a", False),
    )
    assert pool_vote(candidates, 1.5).key == "a"  # 2.5 vs 2.0
    assert pool_vote(candidates, 1.0).key == "a"  # tie on weight, "a" has the leader vote


def test_ties_break_by_leader_votes_then_support():
    candidates = pool((ROUND, 1, 0, "a", True), (ROUND, 1, 1, "b", False), (ROUND, 1, 2, "b", False))
    vote = pool_vote(candidates, 2.0)  # a: 2.0 with 1 leader vote, b: 2.0 with 0
    assert vote.key == "a" and vote.leader_support == 1


def test_representative_prefers_leader_then_latest_stage_then_lowest_agent():
    candidates = pool(
        (INITIAL, 0, 0, "a", False),
        (REVIEW, 0, 1, "a", False),
        (ROUND, 2, 3, "a", False),
        (ROUND, 2, 2, "a", False),
    )
    assert pool_vote(candidates, 1.5).representative.agent == 2
    candidates = pool((ROUND, 3, 0, "a", False), (INITIAL, 0, 1, "a", True))
    assert pool_vote(candidates, 1.5).representative.source == INITIAL
    assert priority(candidates[0]) < priority(candidates[1])


def test_empty_keys_are_ignored():
    assert pool_vote(pool((INITIAL, 0, 0, "", True)), 1.5) is None


def test_best_index_prefers_nonempty_responses():
    assert best_index(["", "x", "y"], [2.0, 1.0, 1.0]) == 1
    assert best_index(["", ""], [0.0, 1.0]) == 1
