"""Weighted pool vote over all reasoning stages (paper Eq. 9).

The pool holds the initial answers, one kept review per reviewed agent, and every
completed collaboration round. Each occurrence votes for its answer key with
weight 1 (1 + β for the stage leader). Ties are broken by leader votes, then total
votes. The returned response prefers a leader's, then the most recent stage's,
then the lowest agent index's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

INITIAL, REVIEW, ROUND = "initial", "review", "round"


@dataclass(frozen=True)
class Candidate:
    source: str  # INITIAL, REVIEW or ROUND
    round: int  # 0 for initial answers and reviews, t for collaboration round t
    agent: int
    text: str
    key: str
    is_leader: bool  # the stage leader (the donor for reviews)
    index: int  # position in the pool


@dataclass(frozen=True)
class VoteResult:
    key: str
    weight: float
    support: int
    leader_support: int
    representative: Candidate


def best_index(texts: Sequence[str], scores: Sequence[float]) -> int:
    """The stage leader: highest score among non-empty responses, lowest index on ties."""
    if not texts:
        return 0
    nonempty = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
    return max(nonempty or range(len(texts)), key=lambda i: (scores[i], -i))


def priority(candidate: Candidate) -> tuple:
    if candidate.source == ROUND:
        stage = 30 + candidate.round
    elif candidate.source == REVIEW:
        stage = 20
    else:
        stage = 10
    return (candidate.is_leader, stage, -candidate.agent, -candidate.index)


def pool_vote(candidates: Sequence[Candidate], leader_weight: float) -> VoteResult | None:
    """Weighted vote over non-empty answer keys; ``None`` when no candidate has an answer."""
    groups: dict[str, dict] = {}
    for candidate in candidates:
        if not candidate.key:
            continue
        group = groups.setdefault(
            candidate.key, {"weight": 0.0, "support": 0, "leader_support": 0, "representative": candidate}
        )
        group["weight"] += leader_weight if candidate.is_leader else 1.0
        group["support"] += 1
        group["leader_support"] += int(candidate.is_leader)
        if priority(candidate) > priority(group["representative"]):
            group["representative"] = candidate
    if not groups:
        return None
    key, group = max(
        groups.items(),
        key=lambda item: (
            item[1]["weight"],
            item[1]["leader_support"],
            item[1]["support"],
            priority(item[1]["representative"]),
        ),
    )
    return VoteResult(key, group["weight"], group["support"], group["leader_support"], group["representative"])
