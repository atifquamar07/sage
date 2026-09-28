"""SAGE: Self-Adapting Group of Experts (Algorithm 1 of the paper).

For each question:

1. **Select a donor.** Agents answer independently. Responses are scored by answer
   agreement and prefix consistency, and reciprocal peer review picks the donor.
2. **Transfer strategy.** Every other agent rewrites its role prompt with guidance
   from the donor's original prompt.
3. **Collaborate.** For up to T rounds, agents revise along a sparse DAG in which
   each agent reads at most K strictly higher-scoring parents. The graph is rebuilt
   from the new scores after every round; collaboration stops early on consensus.
4. **Pool and vote.** A weighted vote over all stages gives the final answer.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field

from .answer.extractor import AnswerExtractor
from .config import SageConfig
from .context import QuestionContext
from .core.review import ReviewResult, peer_review
from .core.rewrite import rewrite_prompts
from .core.routing import is_consensus, parents, score_edges, update_order
from .core.scoring import Score, score_answers
from .core.vote import INITIAL, REVIEW, ROUND, Candidate, best_index, pool_vote
from .llm.client import LLM
from .llm.trace import CallRecord, usage_totals
from .prompts.roles import role_prompt, sample_roles
from .prompts.templates import initial_prompt, leader_prompt, update_prompt
from .seeding import derive_seed


@dataclass
class Round:
    round: int
    edges: list[tuple[int, int]]
    order: list[int]
    answers: list[str]
    scores: list[Score]
    errors: dict[int, str] = field(default_factory=dict)


@dataclass
class SageResult:
    answer: str  # final response; its last line holds the boxed answer when one exists
    answer_key: str  # normalized key of the voted answer
    donor: int | None
    trace: dict
    calls: list[CallRecord]

    def to_dict(self, full_prompts: bool = False) -> dict:
        return {
            "answer": self.answer,
            "answer_key": self.answer_key,
            "donor": self.donor,
            "trace": self.trace,
            "usage": usage_totals(self.calls),
            "calls": [call.to_dict(full_prompts) for call in self.calls],
        }


class SAGE:
    """Run SAGE on one question at a time; the instance is thread-safe.

    Args:
        config: method hyperparameters (defaults follow the paper: N=4, K=2, T=3).
        llm: chat-completion backend, e.g. :class:`~sage_mas.llm.client.OpenAICompatClient`.
        extractor: answer extractor κ, e.g. :class:`~sage_mas.answer.extractor.XFinderExtractor`.
        model: model name used by every agent unless ``config.backbones`` assigns one per agent.
    """

    def __init__(self, config: SageConfig, llm: LLM, extractor: AnswerExtractor, model: str | None = None):
        self.config = config.validate()
        backbones = config.backbones or ([model] * config.num_agents if model else None)
        if not backbones:
            raise ValueError("pass model= or set config.backbones")
        self.backbones = list(backbones)
        self.llm = llm
        self.extractor = extractor

    def run(self, sample: Mapping, seed: int, *, review_rng: random.Random | None = None) -> SageResult:
        """Answer ``sample`` (a dict with ``query`` and optional ``choices``).

        ``seed`` fixes the role sampling, the reviewer sampling and the decoding seed.
        ``review_rng`` overrides the reviewer-sampling generator (used to replay
        recorded runs).
        """
        cfg = self.config
        roles = sample_roles(sample, seed, cfg.roles, cfg.num_agents)
        question_id = sample.get("sample_id") or sample.get("query")
        ctx = QuestionContext(
            query=str(sample["query"]),
            choices=sample.get("choices") or [],
            seed=seed,
            config=cfg,
            llm=self.llm,
            extractor=self.extractor,
            backbones=self.backbones,
            roles=roles,
            system_prompts=[role_prompt(role) for role in roles],
            review_rng=review_rng or random.Random(derive_seed(seed, "review", question_id)),
        )
        n = cfg.num_agents

        # Stage 1: independent answers, scores ρ⁰, reciprocal peer review, donor.
        initial = ctx.parallel(
            lambda i: ctx.call(
                agent=i,
                stage="initial_generation",
                round=0,
                prompt=initial_prompt(ctx.query),
                system=ctx.system_prompts[i],
                temperature=cfg.temperature,
            ),
            range(n),
        )
        initial_scores = score_answers(ctx, initial)
        rho0 = [s.value for s in initial_scores]
        review = peer_review(ctx, initial, rho0)
        donor = review.donor if review.donor is not None else max(range(n), key=lambda i: (rho0[i], -i))

        # Stage 2: strategy transfer from the donor's original prompt.
        original_prompts = list(ctx.system_prompts)
        ctx.system_prompts, rewrites = rewrite_prompts(ctx, original_prompts, donor)
        ctx.prefix_cache.clear()  # later prefix checks use the adapted prompts

        # Stage 3: collaboration on the score-directed DAG, starting from Y⁰ and ρ⁰.
        rounds = self._collaborate(ctx, initial, rho0)

        # Stage 4: weighted vote over the pooled stages.
        final = rounds[-1]
        final_values = [s.value for s in final.scores]
        best_final = best_index(final.answers, final_values)
        candidates = self._pool(ctx, initial, rho0, review, rounds)
        vote = pool_vote(candidates, leader_weight=1.0 + cfg.vote.leader_bonus)
        answer = ctx.ensure_boxed(vote.representative.text) if vote else ""
        answer_key = vote.key if vote else ""
        if not answer:  # no pooled response has an answer: fall back to the final leader
            answer = ctx.ensure_boxed(final.answers[best_final])
            answer_key = ctx.key(answer)

        trace = _trace(
            roles, original_prompts, initial, initial_scores, review, donor, rewrites, rounds, vote, best_final
        )
        return SageResult(answer, answer_key, donor, trace, list(ctx.calls))

    def _collaborate(self, ctx: QuestionContext, initial: list[str], rho0: list[float]) -> list[Round]:
        cfg = self.config
        current = list(initial)
        values = list(rho0)
        edges = score_edges(values, cfg.routing.top_k)
        rounds: list[Round] = []
        for t in range(1, cfg.routing.max_rounds + 1):
            order = update_order(values)
            errors = {}
            for agent in order:
                agent_parents = parents(edges, agent)
                if not agent_parents and agent != order[0]:
                    continue  # sources other than the leader keep their answer
                if agent_parents:
                    prompt = update_prompt(ctx.query, current[agent], [(p, current[p]) for p in agent_parents])
                else:
                    prompt = leader_prompt(ctx.query, current[agent])
                try:
                    current[agent] = ctx.call(
                        agent=agent,
                        stage="collaboration",
                        round=t,
                        prompt=prompt,
                        system=ctx.system_prompts[agent],
                        temperature=cfg.temperature,
                    )
                except Exception as exc:  # keep the previous answer
                    errors[agent] = f"{type(exc).__name__}: {exc}"
            scores = score_answers(ctx, current)
            rounds.append(Round(t, edges, order, list(current), scores, errors))
            if is_consensus([s.key for s in scores]):
                break
            values = [s.value for s in scores]
            edges = score_edges(values, cfg.routing.top_k)
        return rounds

    @staticmethod
    def _pool(
        ctx: QuestionContext, initial: list[str], rho0: list[float], review: ReviewResult, rounds: list[Round]
    ) -> list[Candidate]:
        pool: list[Candidate] = []

        def add(source: str, round_: int, agent: int, text: str, is_leader: bool) -> None:
            pool.append(Candidate(source, round_, agent, text, ctx.key(text), is_leader, len(pool)))

        leader = best_index(initial, rho0)
        for agent, text in enumerate(initial):
            add(INITIAL, 0, agent, text, agent == leader)
        for kept in review.retained:
            add(REVIEW, 0, kept.agent, kept.final_answer, kept.agent == review.donor)
        for rnd in rounds:
            leader = best_index(rnd.answers, [s.value for s in rnd.scores])
            for agent, text in enumerate(rnd.answers):
                add(ROUND, rnd.round, agent, text, agent == leader)
        return pool


def _scores(scores: list[Score]) -> list[dict]:
    return [
        {"key": s.key, "agreement": s.agreement, "prefix_consistent": s.prefix_consistent, "score": s.value}
        for s in scores
    ]


def _trace(roles, prompts, initial, initial_scores, review, donor, rewrites, rounds, vote, best_final) -> dict:
    return {
        "roles": roles,
        "initial": {"answers": initial, "scores": _scores(initial_scores)},
        "review": {
            "reason": review.reason,
            "high": review.high,
            "low": review.low,
            "sampled": [review.sampled_high, review.sampled_low],
            "reviews": [
                {
                    "owner": r.owner,
                    "reviewed": r.reviewed,
                    "action": r.action,
                    "final_answer": r.final_answer,
                    "error": r.error,
                    **_scores([s])[0],
                }
                for r, s in zip(review.reviews, review.candidate_scores, strict=True)
            ],
            "retained": [
                {"agent": k.agent, "action": k.action, "candidate": k.candidate, **_scores([s])[0]}
                for k, s in zip(review.retained, review.retained_scores, strict=True)
            ],
        },
        "donor": donor,
        "rewrites": [
            {
                "agent": log.agent,
                "status": log.status,
                "prompt": log.prompt if log.prompt != prompts[log.agent] else None,
                "attempts": log.attempts,
            }
            for log in rewrites
        ],
        "rounds": [
            {
                "round": r.round,
                "edges": [list(edge) for edge in r.edges],
                "order": r.order,
                "answers": r.answers,
                "scores": _scores(r.scores),
                "errors": {str(k): v for k, v in r.errors.items()},
            }
            for r in rounds
        ],
        "vote": None
        if vote is None
        else {
            "key": vote.key,
            "weight": vote.weight,
            "support": vote.support,
            "leader_support": vote.leader_support,
            "source": vote.representative.source,
            "round": vote.representative.round,
            "agent": vote.representative.agent,
        },
        "final_leader": best_final,
    }
