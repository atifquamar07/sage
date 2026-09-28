"""Helpers for ``\\boxed{...}`` final answers."""

from __future__ import annotations

import re
from collections.abc import Callable

BOXED_RE = re.compile(r"(?<![A-Za-z])\\?boxed\s*\{")

# Boxed contents that are format placeholders rather than answers.
_PLACEHOLDERS = frozenset({"", "...", "\\ldots", "\\dots", "\\cdots", "?", "answer", "finalanswer"})


def repair(text: object) -> str:
    """Strip and repair ``\\b`` escapes that turned ``\\boxed`` into ``<BS>oxed``."""
    return str(text or "").strip().replace("\x08oxed", r"\boxed")


def balanced_braces(text: str, open_idx: int) -> str | None:
    """Return the contents of the brace group opening at ``open_idx``."""
    if open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    escaped = False
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx]
    return None


def is_placeholder(content: object) -> bool:
    return re.sub(r"\s+", "", str(content or "").strip().lower()) in _PLACEHOLDERS


def last_boxed_answer(text: object) -> str:
    """Return the last non-placeholder ``\\boxed{}`` content, or ``""``."""
    text = repair(text)
    found = []
    for match in BOXED_RE.finditer(text):
        content = balanced_braces(text, text.find("{", match.end() - 1))
        if content is None:
            continue
        content = content.strip()
        if not is_placeholder(content):
            found.append(content)
    return found[-1] if found else ""


def ends_with_boxed_answer(text: object) -> bool:
    lines = [line.strip() for line in repair(text).splitlines() if line.strip()]
    if not lines:
        return False
    return bool(BOXED_RE.search(lines[-1]) and last_boxed_answer(lines[-1]))


def ensure_boxed(text: object, extract: Callable[[str], str | None]) -> str:
    """Make a response end with a boxed final answer when one can be extracted.

    If the last non-empty line already holds a boxed answer the text is returned
    as is; otherwise ``extract`` (the active answer extractor) is asked for the
    final answer, which is appended as ``Final answer: \\boxed{...}``.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return stripped
    repaired = stripped.replace("\x08oxed", r"\boxed")
    if ends_with_boxed_answer(repaired):
        return repaired
    extracted = extract(repaired)
    if extracted:
        return f"{repaired.rstrip()}\n\nFinal answer: \\boxed{{{extracted}}}"
    return repaired
