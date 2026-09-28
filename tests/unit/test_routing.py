"""Score-directed DAG: Eq. 7 and Lemma 2.1 (acyclic, in-degree ≤ K, |E| bound, depth ≤ K)."""

import heapq
import itertools
from collections import defaultdict

import pytest

from sage_mas.core.routing import is_consensus, parents, score_edges, update_order

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


def _longest_path(n, edges):
    children = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
    memo = {}

    def depth(u):
        if u not in memo:
            memo[u] = max((1 + depth(v) for v in children[u]), default=0)
        return memo[u]

    return max((depth(u) for u in range(n)), default=0)


def _kahn_order(scores, edges):
    """Topological order preferring higher score, then lower index."""
    n = len(scores)
    indeg, children = [0] * n, defaultdict(list)
    for u, v in edges:
        children[u].append(v)
        indeg[v] += 1
    heap = [(-scores[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (-scores[v], v))
    return order


@pytest.mark.parametrize("n,k", [(n, k) for n in range(2, 6) for k in range(0, min(n, 4))])
def test_lemma_holds_for_all_score_vectors(n, k):
    for scores in itertools.product(LEVELS[: n + 1], repeat=n):
        edges = score_edges(scores, k)
        assert all(scores[u] > scores[v] for u, v in edges)  # strictly decreasing => acyclic
        indegree = defaultdict(int)
        for _, v in edges:
            indegree[v] += 1
        assert max(indegree.values(), default=0) <= k
        assert len(edges) <= k * n - k * (k + 1) // 2
        assert _longest_path(n, edges) <= k
        assert update_order(scores) == _kahn_order(scores, edges)


def test_parents_are_top_k_strictly_higher_with_index_tie_break():
    scores = [1.0, 1.5, 1.5, 0.5]
    edges = score_edges(scores, 2)
    assert parents(edges, 3) == [1, 2]
    assert parents(edges, 0) == [1, 2]
    assert parents(edges, 1) == [] and parents(edges, 2) == []
    assert parents(score_edges([1.0, 1.5, 1.5, 1.25], 1), 3) == [1]  # equal scores: lower index wins


def test_equal_scores_create_no_edges():
    assert score_edges([0.75] * 4, 2) == []
    assert update_order([0.75] * 4) == [0, 1, 2, 3]


def test_consensus():
    assert is_consensus(["num:4", "num:4"])
    assert not is_consensus(["num:4", "num:5"])
    assert not is_consensus(["", ""])
    assert not is_consensus([])
