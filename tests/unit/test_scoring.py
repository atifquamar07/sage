"""ρ = q + λ·z: answer agreement and prefix consistency (Eq. 1-3)."""

from conftest import make_context

from sage_mas.config import PrefixConfig, SageConfig
from sage_mas.core.scoring import answer_prefix, score_answers


def test_answer_prefix():
    text = "one two  three four five\nsix seven eight nine ten"
    assert answer_prefix(text, 0.6) == "one two  three four five\nsix "
    assert answer_prefix("single", 0.6) == "single"  # at least one token
    assert answer_prefix("", 0.6) == ""
    assert answer_prefix("a b", 0.1) == "a "


def test_agreement_and_prefix_consistency():
    # Prefix completions reproduce the agent's own answer, except agent 3's.
    def respond(request):
        return "so \\boxed{9}" if request.agent == 3 else "done \\boxed{%s}" % ("4" if request.agent < 2 else "5")

    ctx = make_context(respond)
    texts = ["a b c d \\boxed{4}", "e f g h \\boxed{4}", "i j k l \\boxed{5}", "m n o p \\boxed{5}"]
    scores = score_answers(ctx, texts)
    assert [s.key for s in scores] == ["num:4", "num:4", "num:5", "num:5"]
    assert [s.agreement for s in scores] == [0.5, 0.5, 0.5, 0.5]
    assert [s.prefix_consistent for s in scores] == [True, True, True, False]
    assert [s.value for s in scores] == [1.0, 1.0, 1.0, 0.5]
    assert all(call.request.stage == "prefix_continuation" for call in ctx.calls)


def test_prefix_checks_are_cached_and_reset():
    ctx = make_context(lambda r: "\\boxed{4}")
    texts = ["a b c \\boxed{4}"] * 2
    score_answers(ctx, texts, agents=[0, 0])
    score_answers(ctx, texts, agents=[0, 0])
    assert len(ctx.calls) == 1  # one check per (agent, text)
    ctx.prefix_cache.clear()
    score_answers(ctx, texts[:1], agents=[0])
    assert len(ctx.calls) == 2


def test_no_answer_scores_zero_without_calls():
    ctx = make_context(lambda r: "\\boxed{4}")
    scores = score_answers(ctx, ["no answer here", "a b c \\boxed{4}"])
    assert scores[0].key == "" and scores[0].value == 0.0
    assert [c.request.agent for c in ctx.calls] == [1]


def test_lambda_zero_disables_prefix_calls():
    ctx = make_context(config=SageConfig(prefix=PrefixConfig(weight=0.0)))
    scores = score_answers(ctx, ["a b c \\boxed{4}", "\\boxed{4}"])
    assert [s.value for s in scores] == [1.0, 1.0] and not ctx.calls


def test_prefix_failure_counts_as_inconsistent():
    def respond(request):
        raise RuntimeError("server down")

    ctx = make_context(respond)
    scores = score_answers(ctx, ["a b c \\boxed{4}"])
    assert scores[0].value == 1.0 and not scores[0].prefix_consistent
