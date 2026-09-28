"""Reciprocal peer review and donor selection (paper Eq. 4-5, Appendix D).

1. Split the agents into a higher- and a lower-scoring group by ρ⁰ and sample up
   to m agents from each.
2. For every cross-group pair (i, j), both agents review: each reads its own and
   the other's initial response and keeps (KEEP) or revises (EDIT) its answer.
3. Score all review candidates jointly and keep each reviewed agent's best one.
4. Rescore the kept candidates among themselves; the best agent is the donor.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..context import QuestionContext
from ..prompts.templates import CRITIC_SYSTEM, critic_user_prompt
from .scoring import Score, score_answers

_ACTION_RE = re.compile(r"(?im)^\s*action\s*:\s*`?(keep|edit)`?\s*$")
_FINAL_ANSWER_RE = re.compile(r"(?im)^\s*final_answer\s*:\s*")


@dataclass(frozen=True)
class Review:
    pair: int
    owner: int  # the reviewing agent, whose answer this is
    reviewed: int  # the agent whose response was read
    action: str  # "KEEP" or "EDIT"
    final_answer: str
    raw: str
    error: str | None  # set when the call or parsing failed (then KEEP)


@dataclass(frozen=True)
class Retained:
    agent: int
    action: str
    final_answer: str
    candidate: int | None  # index into the reviews; None if the agent had no pair


@dataclass
class ReviewResult:
    reason: str
    high: list[int]
    low: list[int]
    sampled_high: list[int]
    sampled_low: list[int]
    reviews: list[Review]
    candidate_scores: list[Score]
    retained: list[Retained]
    retained_scores: list[Score]
    donor: int | None


def split_groups(scores: Sequence[float], rng: random.Random) -> tuple[list[int], list[int], str]:
    """Higher-scoring group: the top-scoring agents, plus the next best agent when only one
    agent has the top score. If all scores are equal, two agents are drawn at random."""
    agents = list(range(len(scores)))
    if not agents:
        return [], [], "no_agents"
    if len(set(scores)) == 1:
        chosen = set(rng.sample(agents, min(2, len(agents))))
        return (
            [a for a in agents if a in chosen],
            [a for a in agents if a not in chosen],
            "all_scores_equal_random_top_two",
        )
    top = max(scores)
    high = [a for a in agents if scores[a] == top]
    if len(high) == 1 and len(agents) > 1:
        high.append(max((a for a in agents if a not in high), key=lambda a: (scores[a], -a)))
    return high, [a for a in agents if a not in high], "top_score_plus_next_best_if_needed"


def sample_group(group: Sequence[int], m: int, rng: random.Random) -> list[int]:
    size = min(m, len(group))
    if size <= 0:
        return []
    if len(group) <= size:
        return list(group)
    return sorted(rng.sample(list(group), size))


def parse_critique(text: str) -> tuple[str, str]:
    """Parse ``action: KEEP|EDIT`` and the ``final_answer:`` block of a review."""
    text = text if isinstance(text, str) else ""
    actions = list(_ACTION_RE.finditer(text))
    finals = list(_FINAL_ANSWER_RE.finditer(text))
    if len(actions) != 1:
        raise ValueError("critique response requires exactly one explicit KEEP/EDIT action")
    if len(finals) != 1:
        raise ValueError("critique response requires exactly one final_answer block")
    if actions[0].start() > finals[0].start():
        raise ValueError("critique action must precede final_answer")
    final_answer = text[finals[0].end() :].strip()
    if not final_answer or final_answer.casefold() in {"```", "```text"}:
        raise ValueError("critique final_answer block is empty")
    return actions[0].group(1).upper(), final_answer


def _review(ctx: QuestionContext, answers: Sequence[str], pair: int, own: int, other: int) -> Review:
    raw = ""
    try:
        raw = ctx.call(
            agent=own,
            stage="reciprocal_comparison",
            round=0,
            prompt=critic_user_prompt(
                ctx.query, own, ctx.roles[own], answers[own], other, ctx.roles[other], answers[other]
            ),
            system=CRITIC_SYSTEM,
            temperature=ctx.config.temperature,
        )
        action, final_answer = parse_critique(raw)
        return Review(pair, own, other, action, ctx.ensure_boxed(final_answer), raw, None)
    except Exception as exc:
        return Review(pair, own, other, "KEEP", ctx.ensure_boxed(answers[own]), raw, f"{type(exc).__name__}: {exc}")


def peer_review(ctx: QuestionContext, answers: Sequence[str], scores: Sequence[float]) -> ReviewResult:
    """Run reciprocal peer review on the initial ``answers`` with scores ρ⁰ and pick the donor."""
    rng = ctx.review_rng
    high, low, reason = split_groups(scores, rng)
    m = ctx.config.review.sample_size
    sampled_high = sample_group(high, m, rng)
    sampled_low = sample_group(low, m, rng)

    tasks = []
    for pair, (i, j) in enumerate((i, j) for i in sampled_high for j in sampled_low):
        tasks += [(pair, i, j), (pair, j, i)]
    reviews = ctx.parallel(lambda task: _review(ctx, answers, *task), tasks)

    # Score all candidates jointly; keep each reviewed agent's best candidate.
    candidate_scores = score_answers(ctx, [r.final_answer for r in reviews], [r.owner for r in reviews])
    retained = []
    for agent in sampled_high + sampled_low:
        own = [idx for idx, review in enumerate(reviews) if review.owner == agent]
        if not own:
            retained.append(Retained(agent, "KEEP", answers[agent], None))
            continue
        best = max(
            own,
            key=lambda idx: (candidate_scores[idx].value, reviews[idx].action == "EDIT", -idx),
        )
        retained.append(Retained(agent, reviews[best].action, reviews[best].final_answer, best))

    # Rescore the kept candidates among themselves; the best agent is the donor.
    retained_scores = score_answers(ctx, [r.final_answer for r in retained], [r.agent for r in retained])
    donor = None
    if retained:
        best = max(
            range(len(retained)),
            key=lambda idx: (retained_scores[idx].value, scores[retained[idx].agent], -retained[idx].agent),
        )
        donor = retained[best].agent

    return ReviewResult(
        reason=reason,
        high=high,
        low=low,
        sampled_high=sampled_high,
        sampled_low=sampled_low,
        reviews=reviews,
        candidate_scores=candidate_scores,
        retained=retained,
        retained_scores=retained_scores,
        donor=donor,
    )
