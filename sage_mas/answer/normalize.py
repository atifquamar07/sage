"""Answer normalization and answer keys (the paper's normalized answer κ).

Two responses agree when their extracted answers map to the same non-empty key:
``num:<exact rational>`` for numbers, ``choice:<letter>`` for option letters and
``text:<normalized string>`` otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation
from fractions import Fraction

BOXED_RE = re.compile(r"(?<![A-Za-z])\\?boxed\s*\{")
FULL_NUMBER_RE = re.compile(
    r"^[-+]?\$?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)?%?$"
)
_CHOICE_RE = re.compile(r"^(?:\*\*)?\(?(?:\*\*)?([A-Ea-e])(?:\*\*)?\)?(?:[\).:-]|\s*(?:\*\*))")


def answer_key(answer: str | None) -> str:
    """Map an extracted answer to its comparison key (``""`` when there is none)."""
    if not answer:
        return ""
    numeric = normalize_numeric_answer(answer)
    if numeric is not None:
        return f"num:{numeric}"
    choice = normalize_choice_answer(answer)
    if choice is not None:
        return f"choice:{choice}"
    normalized = normalize_answer(answer) or ""
    return f"text:{normalized}" if normalized else ""


def repair_boxed_text(text: object) -> str:
    return str("" if text is None else text).replace("\x08oxed", r"\boxed")


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    text = repair_boxed_text(answer).strip()
    if not text:
        return None
    text = strip_answer_prefix(text)
    text = _strip_latex_wrappers(text)
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("$", "").strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\(?:sqrt)\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = re.sub(r"\\(?:cdot|times)", "*", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .")
    text = text.strip("()[]")
    return text.lower() or None


def normalize_numeric_answer(answer: str | None) -> Fraction | None:
    text = normalize_answer(answer)
    if text is None:
        return None
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "").replace(",", "").strip()
    if FULL_NUMBER_RE.fullmatch(text) is None:
        return None
    number = text.replace("$", "").replace(",", "").replace(" ", "")
    is_percent = number.endswith("%")
    if is_percent:
        number = number[:-1]
    try:
        if "/" in number:
            numerator, denominator = number.split("/", 1)
            value = _fraction(numerator) / _fraction(denominator)
        else:
            value = _fraction(number)
        if is_percent:
            value /= 100
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value


def normalize_choice_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    match = _CHOICE_RE.match(str(answer).strip())
    if match is not None:
        return match.group(1).lower()
    normalized = normalize_answer(answer)
    if normalized is None:
        return None
    match = re.fullmatch(r"(?:option)?([a-e])", normalized)
    return match.group(1) if match is not None else None


def strip_answer_prefix(text: str) -> str:
    text = str(text).strip()
    text = re.sub(
        r"^(?:therefore,\s*)?(?:the\s+)?(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^(?:option|choice)\s*\(?([A-E])\)?\s*(?:is|:)?\s*", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def iter_boxed_answers(text: str) -> Iterator[str]:
    text = repair_boxed_text(text)
    for match in BOXED_RE.finditer(text):
        content = _balanced_braces(text, match.end() - 1)
        if content is not None:
            yield content


def _strip_latex_wrappers(text: str) -> str:
    text = str(text).strip()
    changed = True
    while changed:
        changed = False
        boxed = next(iter_boxed_answers(text), None)
        if boxed is not None and boxed.strip() != text:
            text = boxed.strip()
            changed = True
            continue
        for left, right in ((r"\(", r"\)"), (r"\[", r"\]")):
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                changed = True
                break
    return text


def _fraction(text: str) -> Fraction:
    return Fraction(Decimal(text))


def _balanced_braces(text: str, open_idx: int) -> str | None:
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
