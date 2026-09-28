"""Per-question state shared by the SAGE stages.

A :class:`QuestionContext` is created for every question, so a single :class:`SAGE`
instance can serve many questions concurrently.
"""

from __future__ import annotations

import concurrent.futures
import random
import threading
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

from .answer.boxed import ensure_boxed
from .answer.extractor import AnswerExtractor, Extraction
from .config import SageConfig
from .llm.client import LLM, ChatRequest
from .llm.trace import CallRecord

T = TypeVar("T")
R = TypeVar("R")


class QuestionContext:
    def __init__(
        self,
        *,
        query: str,
        choices: Sequence[object],
        seed: int,
        config: SageConfig,
        llm: LLM,
        extractor: AnswerExtractor,
        backbones: Sequence[str],
        roles: Sequence[str],
        system_prompts: Sequence[str],
        review_rng: random.Random,
    ):
        self.query = query
        self.choices = list(choices or [])
        self.seed = seed
        self.config = config
        self.llm = llm
        self.extractor = extractor
        self.backbones = list(backbones)
        self.roles = list(roles)
        # The prompts in use: the original role prompts until strategy transfer
        # replaces them with the adapted prompts.
        self.system_prompts = list(system_prompts)
        self.review_rng = review_rng
        self.calls: list[CallRecord] = []
        self._extractions: dict[str, Extraction] = {}
        self.prefix_cache: dict[tuple[int, str], dict] = {}
        self._lock = threading.Lock()

    # -- model calls ---------------------------------------------------------

    def call(
        self,
        *,
        agent: int,
        stage: str,
        round: int | None,
        prompt: str,
        system: str,
        temperature: float,
        attempt: int | None = None,
    ) -> str:
        """Call agent ``agent``'s backbone and record the call. Errors propagate."""
        request = ChatRequest(
            model=self.backbones[agent],
            system=system,
            prompt=prompt,
            temperature=temperature,
            seed=self.seed,
            stage=stage,
            agent=agent,
            round=round,
            attempt=attempt,
        )
        started = time.monotonic()
        completion, error = None, None
        try:
            completion = self.llm.complete(request)
            return completion.text
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self._lock:
                self.calls.append(CallRecord(len(self.calls), request, completion, error, time.monotonic() - started))

    # -- answers -------------------------------------------------------------

    def extract(self, text: str) -> Extraction:
        """κ(y) for ``text``, cached per question."""
        text = str(text or "")
        with self._lock:
            cached = self._extractions.get(text)
        if cached is not None:
            return cached
        result = self.extractor.extract(self.query, self.choices, text)
        with self._lock:
            return self._extractions.setdefault(text, result)

    def key(self, text: str) -> str:
        return self.extract(text).key

    def ensure_boxed(self, text: str) -> str:
        return ensure_boxed(text, lambda t: self.extract(t).answer)

    # -- helpers -------------------------------------------------------------

    def parallel(self, fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """Apply ``fn`` to ``items`` concurrently; results keep the input order."""
        items = list(items)
        workers = max(1, min(int(self.config.max_workers), len(items)))
        if workers <= 1:
            return [fn(item) for item in items]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))
