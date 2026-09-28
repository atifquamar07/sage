"""Per-call records kept for every model call SAGE makes on a question."""

from __future__ import annotations

from dataclasses import dataclass

from ..seeding import sha256_text
from .client import ChatRequest, Completion


@dataclass
class CallRecord:
    index: int
    request: ChatRequest
    completion: Completion | None
    error: str | None
    latency_seconds: float

    def to_dict(self, full_prompts: bool = False) -> dict:
        request = self.request
        record = {
            "index": self.index,
            "stage": request.stage,
            "agent": request.agent,
            "round": request.round,
            "attempt": request.attempt,
            "model": request.model,
            "temperature": request.temperature,
            "system_sha256": sha256_text(request.system),
            "prompt_sha256": sha256_text(request.prompt),
            "completion": self.completion.text if self.completion else None,
            "prompt_tokens": self.completion.prompt_tokens if self.completion else 0,
            "completion_tokens": self.completion.completion_tokens if self.completion else 0,
            "finish_reason": self.completion.finish_reason if self.completion else None,
            "error": self.error,
            "latency_seconds": round(self.latency_seconds, 3),
        }
        if full_prompts:
            record["system"] = request.system
            record["prompt"] = request.prompt
        return record


def usage_totals(records: list[CallRecord]) -> dict:
    done = [r for r in records if r.completion is not None]
    return {
        "calls": len(done),
        "failed_calls": len(records) - len(done),
        "prompt_tokens": sum(r.completion.prompt_tokens for r in done),
        "completion_tokens": sum(r.completion.completion_tokens for r in done),
    }
