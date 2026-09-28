"""LLM access: one request/response interface, and a client for OpenAI-compatible servers."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ..config import ModelConfig


@dataclass(frozen=True)
class ChatRequest:
    model: str  # model name (key into the configured models)
    system: str | None  # None sends the prompt as the only message
    prompt: str
    temperature: float
    seed: int | None = None
    # Where the call happens in SAGE; recorded in traces and used for replay.
    stage: str = ""
    agent: int | None = None
    round: int | None = None
    attempt: int | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class LLM(Protocol):
    def complete(self, request: ChatRequest) -> Completion: ...


def _retryable(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return True
    return not isinstance(exc, openai.BadRequestError)


class _Endpoint:
    def __init__(self, url: str, api_key: str, timeout: float, max_concurrency: int):
        import openai

        self.client = openai.OpenAI(base_url=url, api_key=api_key or "EMPTY", timeout=timeout)
        self.slots = threading.BoundedSemaphore(max(1, max_concurrency))


class OpenAICompatClient:
    """Chat completions against OpenAI-compatible endpoints (e.g. ``vllm serve``).

    Requests for a model are spread round-robin over its endpoints; each endpoint
    admits at most ``max_concurrency`` simultaneous requests. Decoding settings
    (``max_tokens``, ``top_p``, ``top_k``, ``repetition_penalty``) come from the
    model's :class:`ModelConfig`.
    """

    def __init__(self, models: Mapping[str, ModelConfig]):
        self.models = dict(models)
        self._endpoints = {
            name: [_Endpoint(url, cfg.api_key, cfg.timeout, cfg.max_concurrency) for url in cfg.endpoints]
            for name, cfg in self.models.items()
        }
        self._cycles = {name: itertools.cycle(range(len(eps))) for name, eps in self._endpoints.items()}
        self._lock = threading.Lock()

    def _next_endpoint(self, model: str) -> _Endpoint:
        if model not in self._endpoints:
            raise KeyError(f"model {model!r} is not configured")
        with self._lock:
            return self._endpoints[model][next(self._cycles[model])]

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_retryable),
        reraise=True,
    )
    def complete(self, request: ChatRequest) -> Completion:
        cfg = self.models[request.model]
        endpoint = self._next_endpoint(request.model)
        kwargs: dict = {
            "model": cfg.name,
            "messages": ([{"role": "system", "content": request.system}] if request.system is not None else [])
            + [{"role": "user", "content": request.prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": request.temperature,
        }
        if request.seed is not None:
            kwargs["seed"] = int(request.seed)
        if cfg.sampling.get("top_p") is not None:
            kwargs["top_p"] = float(cfg.sampling["top_p"])
        extra = {key: cfg.sampling[key] for key in ("top_k", "repetition_penalty") if cfg.sampling.get(key) is not None}
        if extra:
            kwargs["extra_body"] = extra
        with endpoint.slots:
            response = endpoint.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content
        if not isinstance(text, str):
            raise ValueError(f"invalid completion from {request.model}: {text!r}")
        usage = response.usage
        return Completion(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None),
        )
