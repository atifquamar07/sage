"""Response scoring: ρ(y, B) = q(y, B) + λ·z_τ(y)  (paper Eq. 1-3).

* q, answer agreement: the fraction of responses in B with the same non-empty answer key.
* z_τ, prefix consistency: the agent completes the first τ fraction of its own
  response and must reach the same answer key again.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..context import QuestionContext
from ..prompts.templates import prefix_prompt


@dataclass(frozen=True)
class Score:
    agent: int
    key: str  # κ(y); "" when no answer could be extracted
    agreement: float  # q(y, B)
    prefix_consistent: bool  # z_τ(y)
    value: float  # ρ(y, B)


def answer_prefix(text: str, tau: float) -> str:
    """The initial fraction ``tau`` of the whitespace-delimited tokens of ``text``."""
    tokens = re.findall(r"\S+\s*", text or "")
    if not tokens:
        return text or ""
    tau = min(1.0, max(0.0, float(tau)))
    keep = math.floor(len(tokens) * tau)
    if tau > 0.0 and keep == 0:
        keep = 1
    return "".join(tokens[:keep])


def prefix_check(ctx: QuestionContext, agent: int, text: str, key: str) -> dict:
    """Prefix consistency of ``text`` for ``agent`` under the agent's current prompt.

    Outcomes are cached per ``(agent, text)``; the cache is cleared when the agents'
    prompts are replaced, so checks are repeated under the adapted prompts.
    """
    cfg = ctx.config.prefix
    if cfg.weight == 0.0:
        return {"consistent": False, "reason": "disabled"}
    if not key:
        return {"consistent": False, "reason": "no_answer"}
    cached = ctx.prefix_cache.get((agent, text))
    if cached is not None:
        return cached

    prefix = answer_prefix(text, cfg.tau)
    error: str | None = None
    if prefix == text:  # nothing left to complete
        completion = text
    else:
        try:
            completion = ctx.call(
                agent=agent,
                stage="prefix_continuation",
                round=None,
                prompt=prefix_prompt(ctx.query, prefix),
                system=ctx.system_prompts[agent],
                temperature=cfg.temperature,
            )
        except Exception as exc:
            completion, error = None, f"{type(exc).__name__}: {exc}"

    if error is not None:
        detail = {"consistent": False, "reason": "regeneration_failed", "error": error}
    else:
        regenerated_key = ctx.key(completion)
        detail = {"consistent": regenerated_key == key, "regenerated_key": regenerated_key}
    ctx.prefix_cache[(agent, text)] = detail
    return detail


def score_answers(
    ctx: QuestionContext,
    texts: Sequence[str],
    agents: Sequence[int] | None = None,
) -> list[Score]:
    """Score each response against the multiset ``texts`` (the paper's ρ(y, B)).

    ``agents[i]`` owns ``texts[i]`` and is the agent that runs its prefix check.
    """
    n = len(texts)
    if n == 0:
        return []
    agents = list(range(n)) if agents is None else list(agents)
    keys = [ctx.key(text) for text in texts]
    counts = Counter(key for key in keys if key)
    weight = ctx.config.prefix.weight
    scores = []
    for text, agent, key in zip(texts, agents, keys, strict=True):
        agreement = counts[key] / float(n) if key else 0.0
        consistent = bool(prefix_check(ctx, agent, text, key)["consistent"])
        scores.append(Score(agent, key, agreement, consistent, agreement + (weight if consistent else 0.0)))
    return scores
