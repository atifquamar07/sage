"""Deterministic (model-free) final-answer extraction.

Used by the ``regex`` answer extractor and to read reference answers. The paper
runs used xFinder for κ during inference; this extractor is the lightweight
alternative that needs no extra model server.
"""

from __future__ import annotations

import re

from .normalize import (
    iter_boxed_answers,
    normalize_answer,
    normalize_choice_answer,
    repair_boxed_text,
    strip_answer_prefix,
)

FINAL_ANSWER_RE = re.compile(
    r"^(?:[#>*\-\s]*)(?:therefore,?\s*|thus,?\s*|so,?\s*)?"
    r"(?:the\s+)?(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)\s*(.+)$",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)?%?"
)
FINAL_LINE_RE = re.compile(r"\b(?:answer|therefore|thus|so|final|result|total|equals?)\b", re.IGNORECASE)
_STEP_HEADER_RE = re.compile(r"\b(?:step|section|part|question|problem)\s*\d+", re.IGNORECASE)
_PLACEHOLDERS = frozenset(
    {"...", "…", "ldots", "\\ldots", "cdots", "\\cdots", "answer", "finalanswer", "theanswer", "youranswer", "final"}
)


def extract_final_answer(text: object) -> str | None:
    """Return the final answer stated in ``text``, or ``None``.

    Prefers the last non-placeholder ``\\boxed{}``; otherwise searches the last
    lines for an explicit answer statement, an option letter or a number.
    """
    text = repair_boxed_text(text)
    boxed = [answer.strip() for answer in iter_boxed_answers(text) if not _is_placeholder(answer)]
    if boxed:
        return boxed[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(reversed(lines)):
        match = FINAL_ANSWER_RE.search(line)
        if match is not None:
            candidate = strip_answer_prefix(match.group(1))
            if normalize_answer(candidate) is not None:
                return candidate
        if normalize_choice_answer(line) is not None:
            return line
        numbers = NUMBER_RE.findall(line)
        if numbers and FINAL_LINE_RE.search(line) and not _STEP_HEADER_RE.search(line):
            return numbers[-1].strip()
        if idx < 3 and numbers:
            return numbers[-1].strip()

    for line in reversed(lines):
        numbers = NUMBER_RE.findall(line)
        if numbers and not _STEP_HEADER_RE.search(line):
            return numbers[-1].strip()

    numbers = NUMBER_RE.findall(text)
    return numbers[-1].strip() if numbers else None


def extract_ground_truth_answer(text: str) -> str:
    """Reference answer from a solution string (GSM8K ``#### answer`` or a stated answer)."""
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    extracted = extract_final_answer(text)
    if extracted is not None and str(extracted).strip():
        return extracted
    return str(text or "").strip()


def _is_placeholder(answer: str) -> bool:
    text = normalize_answer(answer)
    return text is None or text in _PLACEHOLDERS
