"""Strategy transfer by prompt rewriting (paper Eq. 6).

Every agent except the donor rewrites its original system prompt with guidance
from the donor's original prompt, using its own backbone. The rewriter sees only
the two prompts. A rewrite is accepted only if it passes validation: it keeps the
agent's first line (its identity), adds no answers or task content, and differs
materially from the original. Otherwise the request is retried with feedback, and
after the last attempt the agent keeps its original prompt.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..answer.boxed import last_boxed_answer
from ..context import QuestionContext
from ..prompts.templates import (
    REWRITER_SYSTEM,
    first_nonempty_line,
    rewrite_retry_prompt,
    rewrite_user_prompt,
)

ACCEPTED = "accepted"
RETAINED = "retained"  # every attempt failed validation; the original prompt is kept
FAILED = "generation_error"  # a rewrite request failed; the original prompt is kept
DONOR = "donor"  # the donor keeps its original prompt


@dataclass
class RewriteLog:
    agent: int
    status: str
    prompt: str  # the prompt the agent uses from now on
    attempts: list[dict] = field(default_factory=list)


def validate_rewrite(
    raw: str,
    current_prompt: str,
    donor_prompt: str,
    *,
    max_chars: int,
    min_chars: int,
) -> tuple[str, bool, list[str]]:
    """Return ``(prompt, is_valid, error_codes)`` for a rewriter completion."""
    text = str(raw or "").strip()
    errors = []
    lines = text.splitlines()
    if not text:
        errors.append("empty_rewrite")
    elif lines and lines[0].strip().startswith("```"):
        if len(lines) >= 2 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
        else:
            errors.append("unclosed_markdown_fence")
    if "```" in text:
        errors.append("embedded_markdown_fence")
    if re.search(r"</?[A-Za-z][^>]*>", text):
        errors.append("contains_xml_or_html_tag")
    if re.search(r"(?im)^\s*(?:rewritten|improved|refined)\s+system\s+prompt\s*:", text):
        errors.append("contains_rewrite_label")
    if len(text) > max_chars:
        errors.append("rewrite_too_long")
    if len(text) < min_chars:
        errors.append("rewrite_too_short")

    identity = first_nonempty_line(current_prompt)
    donor_identity = first_nonempty_line(donor_prompt)
    if identity and first_nonempty_line(text) != identity:
        errors.append("current_agent_identity_not_preserved")
    if donor_identity and donor_identity != identity and donor_identity in text:
        errors.append("contains_best_agent_identity")
    if last_boxed_answer(text):
        errors.append("contains_nonempty_boxed_answer")
    if re.search(r"(?im)^\s*(?:\*\*)?action(?:\*\*)?\s*:", text):
        errors.append("contains_critique_action_output")
    if re.search(r"(?im)^\s*(?:\*\*)?final_answer(?:\*\*)?\s*:", text):
        errors.append("contains_critique_final_answer_output")
    for match in re.finditer(
        r"(?im)^\s*(?:\*\*)?(?:the\s+)?(?:final\s+answer|answer)(?:\*\*)?\s*(?:is\s*|:\s*)(.+?)\s*$", text
    ):
        stated = match.group(1).strip().rstrip(".").strip()
        placeholder = re.fullmatch(
            r"\\?boxed\s*\{\s*(?:|\.\.\.|…|\\(?:ldots|dots|cdots)|answer|final\s*answer)\s*\}",
            stated,
            re.IGNORECASE,
        )
        if stated and not placeholder:
            errors.append("contains_explicit_final_answer")
            break
    if re.search(r"(?i)\b(?:to solve this problem|for this problem|in the current question)\b", text):
        errors.append("contains_task_specific_solution_language")

    errors = list(dict.fromkeys(errors))
    if errors:
        return str(current_prompt), False, errors
    return text, True, []


def is_material(current_prompt: str, rewritten_prompt: str, threshold: float = 0.98) -> bool:
    """Whether the rewrite changes the prompt body (all lines after the first) enough."""

    def body(prompt: str) -> str:
        lines = [line.strip() for line in str(prompt or "").splitlines() if line.strip()]
        return re.sub(r"\s+", " ", "\n".join(lines[1:] if len(lines) > 1 else lines)).strip().lower()

    current, rewritten = body(current_prompt), body(rewritten_prompt)
    if not current or not rewritten or current == rewritten:
        return False
    return difflib.SequenceMatcher(None, current, rewritten).ratio() < threshold


def rewrite_one(ctx: QuestionContext, agent: int, prompts: Sequence[str], donor: int) -> RewriteLog:
    current, donor_prompt = prompts[agent], prompts[donor]
    if agent == donor:
        return RewriteLog(agent, DONOR, current)
    cfg = ctx.config.rewrite
    base = rewrite_user_prompt(current, donor_prompt)
    attempts: list[dict] = []
    errors: list[str] = []
    for attempt in range(1, max(1, cfg.max_attempts) + 1):
        prompt = (
            base
            if attempt == 1
            else rewrite_retry_prompt(base, errors, first_nonempty_line(current), first_nonempty_line(donor_prompt))
        )
        generation_error: str | None = None
        try:
            raw = ctx.call(
                agent=agent,
                stage="prompt_rewriting",
                round=0,
                prompt=prompt,
                system=REWRITER_SYSTEM,
                temperature=cfg.temperature,
                attempt=attempt,
            ).strip()
        except Exception as exc:
            raw, generation_error = "", f"{type(exc).__name__}: {exc}"
        candidate, valid, errors = validate_rewrite(
            raw, current, donor_prompt, max_chars=cfg.max_chars, min_chars=cfg.min_chars
        )
        material = valid and is_material(current, candidate, cfg.materiality_threshold)
        if valid and not material:
            errors = [*errors, "rewrite_not_material"]
        if generation_error:
            errors = [generation_error, *errors]
        attempts.append({"attempt": attempt, "errors": errors, "generation_error": generation_error, "completion": raw})
        if valid and material and generation_error is None:
            return RewriteLog(agent, ACCEPTED, candidate, attempts)
    status = FAILED if any(a["generation_error"] for a in attempts) else RETAINED
    return RewriteLog(agent, status, current, attempts)


def rewrite_prompts(
    ctx: QuestionContext, prompts: Sequence[str], donor: int | None
) -> tuple[list[str], list[RewriteLog]]:
    """Adapt every non-donor prompt using the donor's prompt; returns the new prompts and logs."""
    if donor is None or not 0 <= int(donor) < len(prompts):
        donor = 0
    logs = ctx.parallel(lambda agent: rewrite_one(ctx, agent, prompts, int(donor)), range(len(prompts)))
    return [log.prompt for log in logs], logs
