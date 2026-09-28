"""Score-directed sparse DAG for collaboration (paper Eq. 7-8 and Lemma 2.1).

Each agent receives messages from at most K agents with a strictly higher score.
Because every edge goes from a higher to a lower score, the graph is acyclic,
and the decreasing-score order (ties: lower index first) is a topological order.
"""

from __future__ import annotations

from collections.abc import Sequence

Edge = tuple[int, int]  # (parent, child)


def score_edges(scores: Sequence[float], top_k: int) -> list[Edge]:
    """``TopK_K`` strictly higher-scoring parents for every agent (Eq. 7)."""
    n = len(scores)
    edges = []
    for child in range(n):
        higher = [j for j in range(n) if j != child and scores[j] > scores[child]]
        higher.sort(key=lambda j: (scores[j], -j), reverse=True)
        edges.extend((parent, child) for parent in higher[:top_k])
    return edges


def parents(edges: Sequence[Edge], child: int) -> list[int]:
    return sorted(parent for parent, target in edges if target == child)


def update_order(scores: Sequence[float]) -> list[int]:
    """Agents by decreasing score, lower index first on ties. The first is the leader."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def is_consensus(keys: Sequence[str]) -> bool:
    """All agents gave the same non-empty answer key."""
    return bool(keys) and all(key and key == keys[0] for key in keys)
