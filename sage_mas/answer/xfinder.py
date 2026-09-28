"""Client for the xFinder answer-extraction model (served by vLLM)."""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from ..third_party import xfinder_prompt as xp

RESPONSE_TAIL_CHARS = 16000


class XFinderError(RuntimeError):
    """The xFinder request failed or returned an unusable completion."""


@dataclass(frozen=True)
class XFinderOutput:
    answer: str  # "" when xFinder reports no valid answer
    raw_output: str


def answer_range(choices: Sequence[object] | None = None) -> str:
    """Native xFinder answer range: option code/content pairs, or free-form math."""
    if choices:
        return json.dumps([[chr(ord("A") + i), str(c)] for i, c in enumerate(choices)], ensure_ascii=False)
    return xp.MATH_ANSWER_RANGE


def format_prompt(query: str, response: str, choices: Sequence[object] | None = None) -> str:
    text = str(response or "")
    if len(text) > RESPONSE_TAIL_CHARS:
        text = text[-RESPONSE_TAIL_CHARS:]
    user = (
        f'Question: """{query or ""!s}"""\n\n'
        f'Output sentences: """{text}"""\n\n'
        f"Answer range: {answer_range(choices)}\n\n"
        "Key extracted answer: "
    )
    return xp.PROMPT_TEMPLATE.format(system_prompt=xp.SYSTEM_PROMPT, input_prompt=user)


def _normalized_output(output: object) -> str:
    if not isinstance(output, str):
        return ""
    text = output.strip()
    for token in xp.STOP_TOKENS:
        text = text.split(token, 1)[0].strip()
    text = re.sub(r"^\s*(?:key\s+extracted\s+answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def parse_output(output: object) -> tuple[str, bool]:
    """Return ``(answer, explicit_no_answer)`` for a raw xFinder completion."""
    text = _normalized_output(output)
    if text.casefold() == xp.NO_VALID_ANSWER.casefold():
        return "", True
    return text, False


class XFinderClient:
    """Round-robin client over one or more OpenAI-compatible ``/v1`` endpoints.

    Transport errors and HTTP 429/5xx are retried with exponential backoff; a
    completion that did not stop normally or cannot be parsed fails immediately.
    """

    def __init__(
        self,
        endpoints: Sequence[str],
        model: str = "xfinder-qwen1505",
        *,
        max_tokens: int = 100,
        timeout: float = 180.0,
        retries: int = 5,
    ):
        if not endpoints:
            raise ValueError("xFinder needs at least one endpoint")
        self.endpoints = [url.rstrip("/") for url in endpoints]
        self.model = model
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._cycle = itertools.cycle(range(len(self.endpoints)))
        self._lock = threading.Lock()

    def extract(self, query: str, response: str, choices: Sequence[object] | None = None) -> XFinderOutput:
        payload = {
            "model": self.model,
            "prompt": format_prompt(query, response, choices),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stop": list(xp.STOP_TOKENS),
        }
        body = json.dumps(payload).encode("utf-8")
        errors = []
        for attempt in range(1, self.retries + 1):
            with self._lock:
                url = f"{self.endpoints[next(self._cycle)]}/completions"
            request = urllib.request.Request(
                url,
                data=body,
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as http_response:
                    result = json.loads(http_response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                errors.append(f"attempt {attempt} via {url}: HTTP {exc.code}")
                if not (exc.code == 429 or 500 <= exc.code <= 599):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                errors.append(f"attempt {attempt} via {url}: {type(exc).__name__}: {exc}")
            else:
                return self._parse(result)
            if attempt < self.retries:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise XFinderError("; ".join(errors) or "xFinder request failed")

    @staticmethod
    def _parse(result: object) -> XFinderOutput:
        try:
            choice = result["choices"][0]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise XFinderError(f"malformed xFinder response: {exc}") from exc
        finish_reason = str(choice.get("finish_reason") or "").strip().lower()
        if finish_reason != "stop":
            raise XFinderError(f"finish_reason must be 'stop', got {finish_reason or 'missing'!r}")
        raw = choice.get("text")
        if not isinstance(raw, str):
            raise XFinderError("xFinder completion text is not a string")
        answer, no_answer = parse_output(raw)
        if not answer and not no_answer:
            raise XFinderError("xFinder completion has no parseable answer")
        return XFinderOutput(answer=answer, raw_output=raw)
