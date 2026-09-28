"""LLMs that need no server: a scripted one for tests and demos, and a trace replayer."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping

from ..seeding import sha256_text
from .client import ChatRequest, Completion


class ScriptedLLM:
    """Answers every request with ``respond(request)``."""

    def __init__(self, respond: Callable[[ChatRequest], str]):
        self.respond = respond

    def complete(self, request: ChatRequest) -> Completion:
        return Completion(text=self.respond(request), finish_reason="stop")


class ReplayMiss(RuntimeError):
    """The requested call does not exist in the recorded trace."""


def replay_key(stage, agent, round_, attempt, system: str, prompt: str) -> tuple:
    return (stage, agent, round_, attempt, sha256_text(system), sha256_text(prompt))


class ReplayLLM:
    """Serves recorded completions for byte-identical requests.

    Each recorded call is keyed by ``(stage, agent, round, attempt, system, prompt)``;
    identical requests are served first-in, first-out. A request that was never
    recorded raises :class:`ReplayMiss`, so a successful replay proves that the code
    issued exactly the recorded prompts.
    """

    def __init__(self, calls: Iterable[Mapping]):
        self._queues: dict[tuple, deque] = defaultdict(deque)
        for call in sorted(calls, key=lambda c: c.get("call_index", 0)):
            key = replay_key(
                call["stage"],
                call["agent"],
                call["round"],
                call.get("attempt"),
                call["system_prompt"],
                call["prompt"],
            )
            self._queues[key].append((call["completion"], float(call["temperature"])))
        self._lock = threading.Lock()
        self.misses: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        key = replay_key(request.stage, request.agent, request.round, request.attempt, request.system, request.prompt)
        with self._lock:
            queue = self._queues.get(key)
            if not queue:
                self.misses.append(request)
                raise ReplayMiss(
                    f"no recorded call for stage={request.stage} agent={request.agent} round={request.round}"
                )
            text, temperature = queue.popleft()
        if abs(temperature - float(request.temperature)) > 1e-9:
            raise AssertionError(f"temperature {request.temperature} != recorded {temperature} for {request.stage}")
        return Completion(text=text, finish_reason="stop")

    def unconsumed(self) -> int:
        return sum(len(queue) for queue in self._queues.values())
