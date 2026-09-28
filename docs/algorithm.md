# SAGE: from the paper to the code

This page maps Algorithm 1 and Equations 1-9 of the paper to the implementation. It
also records the implementation details that the paper leaves implicit. All of them
were used for the paper runs.

## Overview

`SAGE.run(sample, seed)` in [`sage_mas/sage.py`](../sage_mas/sage.py) runs one question.

| Algorithm 1 | Paper | Function |
|---|---|---|
| l.1 Answer | y⁰ᵢ = Mᵢ(x; sᵢ) | `SAGE.run`: `initial_generation` calls with `templates.initial_prompt` |
| l.2 Score | Eq. 1-3 | `core/scoring.score_answers` |
| l.3-5 Peer review, keep best reviews, select donor | Eq. 4-5 | `core/review.peer_review` |
| l.6 Rewrite prompts | Eq. 6 | `core/rewrite.rewrite_prompts` |
| l.7-12 Collaborate | Eq. 7-8 | `SAGE._collaborate` with `core/routing` |
| l.13-14 Pool and vote | Eq. 9 | `SAGE._pool`, `core/vote.pool_vote` |

Every model call goes through `QuestionContext.call`
([`context.py`](../sage_mas/context.py)), which records it. The `stage` names are
`initial_generation`, `prefix_continuation`, `reciprocal_comparison`,
`prompt_rewriting` and `collaboration`.

## Roles

Each question gets N distinct roles, sampled without replacement from the nine-role
pool ([`prompts/roles.py`](../sage_mas/prompts/roles.py)). The sampling seed is a hash
of the run seed and the question's identity fields. The same question therefore
always receives the same team, independent of worker count or question order.

## Answer keys κ(y)

κ(y) is the normalized final answer of a response ([`answer/`](../sage_mas/answer/)).

1. **Extraction.** With `answer_extractor: xfinder` (the paper setting), the xFinder
   model extracts the final answer. It sees the question, the options and the
   response, never the reference answer. With `regex`, the last `\boxed{}` or an
   explicit answer statement is used instead.
2. **Normalization.** `normalize.answer_key` maps the extracted answer to:
   - `num:<exact rational>` for numbers, so `0.5`, `1/2` and `50%` agree;
   - `choice:<letter>` for option letters;
   - `text:<normalized text>` otherwise.
3. **No answer.** An empty key means no answer. It never counts as agreement.

Extractions are cached per question.

## Scoring: ρ = q + λ·z (Eq. 1-3)

`core/scoring.py` computes the score.

- **Agreement** q(y, B) is the fraction of the multiset B with the same non-empty key.
- **Prefix consistency** z(y) works as follows:
  - The agent is given the first ⌊τ·n⌋ whitespace-delimited tokens of its response
    (at least one), under its *current* system prompt, and completes it.
  - z = 1 if the completion reaches the same key.
  - If the prefix is the whole response, no call is made and z = 1.
  - A response without a key gets z = 0 and no call.
- **Caching.** Prefix outcomes are cached per `(agent, response)`. The cache is
  cleared once the prompts have been rewritten, so collaboration rescoring checks
  consistency under the adapted prompts.

## Reciprocal peer review and the donor (Eq. 4-5)

`core/review.py` implements this stage.

- **Groups.** The higher-scoring group holds every agent with the top ρ⁰.
  - If only one agent has it, the next best agent is added (highest ρ⁰, then lower
    index), after the top agent.
  - If all scores are equal, two agents are drawn at random.
- **Sampling.** Up to m agents are sampled from each group (sorted by index). Groups
  of at most m agents are kept as is.
- **Reviews.** Each cross-group pair (i, j) produces two reviews, i←j and j←i. They
  use the fixed critic prompt (`templates.CRITIC_SYSTEM`) on the reviewing agent's
  own backbone.
  - A review must contain exactly one `action: KEEP|EDIT` line, followed by one
    `final_answer:` block.
  - Anything else, including Markdown-bold labels such as `**Action:**`, counts as a
    failed review. The agent then keeps (KEEP) its original answer.
- **Keeping the best review.**
  - All review candidates are scored jointly: agreement over all candidates, and the
    prefix check by the owning agent under its original prompt.
  - Each reviewed agent keeps its best candidate, by (ρ, EDIT before KEEP, earlier
    candidate).
- **Donor.** The kept candidates are rescored among themselves. The donor is the best
  by (ρ, ρ⁰, lower index).
- **Where reviews are used.** Kept reviews only select the donor and join the final
  vote. Collaboration starts from the initial answers and ρ⁰.
- **Randomness.** The reviewer-sampling generator is seeded per question (see
  [Randomness](#randomness)).

## Strategy transfer (Eq. 6)

`core/rewrite.py` implements this stage.

- **The rewrite.** Each non-donor agent rewrites its original prompt on its own
  backbone at temperature 0.2. The rewriter receives `templates.REWRITER_SYSTEM`,
  the target's first line and the two original prompts, and nothing else.
- **Validation.** `validate_rewrite` rejects a rewrite that:
  - is empty or shorter than 80 characters;
  - is wrapped in an unclosed fence or contains fences or markup;
  - changes the first line (the agent's identity);
  - mentions the donor's identity;
  - contains a non-empty `\boxed{}`, a stated answer, critique fields or
    task-specific wording.

  A single enclosing code fence is stripped first.
- **Materiality.** A rewrite must also change the prompt *body* (everything after the
  first line). Their difflib similarity must be below 0.98 after whitespace and
  case normalization.
- **Retries and statuses.** A rejected attempt is retried with targeted feedback
  (`templates.rewrite_retry_prompt`), up to 3 attempts. If none passes, the agent
  keeps its original prompt, with status `retained`.
- **The donor** keeps its own prompt.

## Collaboration on the score-directed DAG (Eq. 7-8, Lemma)

`core/routing.py` and `SAGE._collaborate` implement collaboration.

- **Parents.** Before round t, each agent takes as parents at most K agents with a
  strictly higher score, ranked by (score, lower index). Equal scores create no
  edge. Round 1 uses ρ⁰.
- **Update order.** Agents are updated in decreasing score order, lower index first.
  This is a topological order of the graph. The first agent is the round leader.
  - **Agents with parents** revise from their previous answer and the parents'
    *current-round* answers. Parents are listed by index.
  - **The leader** revises alone (`templates.leader_prompt`).
  - **Other sources** keep their answer.
  - **Failed calls** keep the previous answer.
- **After each round.** All answers are rescored under the adapted prompts. SAGE
  stops if every key is equal and non-empty; otherwise it rebuilds the graph. At
  least one round always runs.

`tests/unit/test_routing.py` checks the Lemma exhaustively: the graph is acyclic,
in-degree is at most K, |E| ≤ KN − K(K+1)/2 and every path has at most K edges.

## Weighted pool vote (Eq. 9)

`core/vote.py` implements the vote.

- **The pool** has one entry per occurrence:
  - the N initial answers;
  - one kept review per reviewed agent;
  - the N answers of every completed round.
- **Stage leaders.** Each stage has one leader:
  - for initial answers and rounds, the highest-scoring non-empty answer (lower index
    on ties);
  - for reviews, the donor.
- **Weights.** Leaders vote with weight 1 + β = 1.5, everyone else with 1.
- **Winner.** The key with the largest weight wins; ties go to more leader votes, then
  more votes.
- **The returned response** is the key's highest-priority occurrence: a leader's, then
  the latest stage (rounds, then reviews, then initial answers), then the lower agent
  index. If its last line holds no boxed answer, `Final answer: \boxed{…}` is
  appended with the extracted answer.
- **Fallback.** If no occurrence has an answer, SAGE returns the final round's leader.

## Randomness

`seed` controls three things:

- the per-question role sampling;
- the per-question reviewer sampling, `random.Random(derive_seed(seed, "review", sample_id))`;
- the `seed` sent with every generation request.

A run is therefore reproducible regardless of worker count, given deterministic
servers.

**Difference from the paper runs.** There, one reviewer-sampling generator was shared
by all questions processed by the same worker thread, so sampling depended on
scheduling. The per-question generator gives the same distribution without the
dependence.
