"""Prompt rewriting (Eq. 6): validation, materiality, retries, and statuses."""

import pytest
from conftest import make_context

from sage_mas.core.rewrite import ACCEPTED, DONOR, RETAINED, is_material, rewrite_prompts, validate_rewrite

CURRENT = "You are an expert in algebra.\n\nSolve equations carefully, simplify each expression, and check the result."
DONOR_PROMPT = "You are an expert in arithmetic.\n\nRecompute every value and track units."
GOOD = (
    "You are an expert in algebra.\n\nSolve equations carefully, simplify each expression, and check the result. "
    "Recompute every intermediate value, track units, and estimate the magnitude of the answer."
)


def validate(raw):
    return validate_rewrite(raw, CURRENT, DONOR_PROMPT, max_chars=6000, min_chars=80)


def test_valid_rewrite():
    assert validate(GOOD) == (GOOD, True, [])
    assert validate(f"```\n{GOOD}\n```")[1]  # a single enclosing fence is stripped


@pytest.mark.parametrize(
    "raw,code",
    [
        ("", "empty_rewrite"),
        ("```\n" + GOOD, "unclosed_markdown_fence"),
        (GOOD + "\n```code```", "embedded_markdown_fence"),
        (GOOD + " <b>x</b>", "contains_xml_or_html_tag"),
        ("You are an expert in algebra.\nRewritten system prompt: " + GOOD, "contains_rewrite_label"),
        (GOOD + "x" * 7000, "rewrite_too_long"),
        ("You are an expert in algebra.\nShort.", "rewrite_too_short"),
        ("You are a helpful solver.\n" + GOOD, "current_agent_identity_not_preserved"),
        (GOOD + "\nYou are an expert in arithmetic.", "contains_best_agent_identity"),
        (GOOD + "\nThe result is \\boxed{12}.", "contains_nonempty_boxed_answer"),
        (GOOD + "\naction: KEEP", "contains_critique_action_output"),
        (GOOD + "\nfinal_answer: 3", "contains_critique_final_answer_output"),
        (GOOD + "\nFinal answer: 42", "contains_explicit_final_answer"),
        (GOOD + " For this problem, add.", "contains_task_specific_solution_language"),
    ],
)
def test_validation_codes(raw, code):
    text, ok, errors = validate(raw)
    assert not ok and code in errors and text == CURRENT


def test_placeholder_final_answer_rule_is_allowed():
    assert validate(GOOD + "\nFinal answer: \\boxed{...}")[1]


def test_materiality():
    assert is_material(CURRENT, GOOD)
    assert not is_material(CURRENT, CURRENT)
    assert not is_material(CURRENT, CURRENT.replace("carefully,", "carefully,  "))  # whitespace only
    assert not is_material(CURRENT, CURRENT.replace("an expert in algebra", "someone else"))  # first line only


def test_rewrite_retries_then_accepts_and_donor_keeps_prompt():
    attempts = {}

    def respond(request):
        attempts[request.agent] = attempts.get(request.agent, 0) + 1
        if request.agent == 1:
            return "nonsense"  # always invalid
        if attempts[request.agent] == 1:
            return request.prompt.split("CURRENT_SYSTEM_PROMPT_BEGIN\n")[1].split("\nCURRENT_SYSTEM_PROMPT_END")[0]
        assert "RETRY_FEEDBACK_BEGIN" in request.prompt and "rewrite_not_material" in request.prompt
        return GOOD.replace("algebra", f"topic {request.agent}")

    ctx = make_context(respond)
    prompts = [CURRENT.replace("algebra", f"topic {i}") for i in range(4)]
    new, logs = rewrite_prompts(ctx, prompts, donor=2)
    assert [log.status for log in logs] == [ACCEPTED, RETAINED, DONOR, ACCEPTED]
    assert new[2] == prompts[2] and new[1] == prompts[1]
    assert new[0] == GOOD.replace("algebra", "topic 0")
    assert [len(log.attempts) for log in logs] == [2, 3, 0, 2]
    assert all(c.request.stage == "prompt_rewriting" and c.request.temperature == 0.2 for c in ctx.calls)
