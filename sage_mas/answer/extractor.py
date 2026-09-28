"""Answer extractors: the function κ(y) that maps a response to its answer key."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .normalize import answer_key
from .regex import extract_final_answer
from .xfinder import XFinderClient, XFinderError


@dataclass(frozen=True)
class Extraction:
    answer: str  # extracted final answer, "" when none was found
    key: str  # normalized answer key κ(y), "" when there is no answer
    error: str | None = None  # set when the extractor itself failed


class AnswerExtractor(Protocol):
    def extract(self, query: str, choices: Sequence[object], text: str) -> Extraction: ...


class XFinderExtractor:
    """κ(y) via the xFinder model, as in the paper. Failures yield an empty key."""

    def __init__(self, client: XFinderClient):
        self.client = client

    def extract(self, query: str, choices: Sequence[object], text: str) -> Extraction:
        try:
            output = self.client.extract(query, text, choices)
        except XFinderError as exc:
            return Extraction(answer="", key="", error=f"{type(exc).__name__}: {exc}")
        return Extraction(answer=output.answer, key=answer_key(output.answer))


class RegexExtractor:
    """κ(y) from the last ``\\boxed{}`` or a stated final answer; needs no model."""

    def extract(self, query: str, choices: Sequence[object], text: str) -> Extraction:
        answer = extract_final_answer(text) or ""
        return Extraction(answer=answer, key=answer_key(answer))
