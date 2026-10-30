"""A deterministic provider for local runs, tests and load tests: it echoes the
last user message and can be told to be slow or to fail."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from ..schemas import ChatRequest, Completion, StreamChunk, Usage
from .base import Provider, ProviderError


def rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockProvider(Provider):
    def __init__(self, name: str, latency_ms: float = 0, failure_rate: float = 0.0, fail_status: int = 503,
                 seed: int | None = None):
        self.name = name
        self.latency = latency_ms / 1000
        self.failure_rate = failure_rate
        self.fail_status = fail_status
        self.rng = random.Random(seed)
        self.calls = 0
        self.fail_next = 0  # tests set this to force the next N calls to fail

    def _maybe_fail(self) -> None:
        self.calls += 1
        retryable = self.fail_status >= 500 or self.fail_status == 429
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ProviderError(f"{self.name}: injected failure", status=self.fail_status, retryable=retryable)
        if self.failure_rate and self.rng.random() < self.failure_rate:
            raise ProviderError(f"{self.name}: random failure", status=self.fail_status, retryable=retryable)

    def _answer(self, req: ChatRequest, model: str) -> tuple[str, Usage]:
        last = next(m.content for m in reversed(req.messages) if m.role == "user")
        text = f"[{self.name}/{model}] {last}"[: req.output_limit(256) * 4]
        prompt = sum(rough_tokens(m.content) for m in req.messages)
        out = rough_tokens(text)
        return text, Usage(prompt_tokens=prompt, completion_tokens=out, total_tokens=prompt + out)

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        if self.latency:
            await asyncio.sleep(self.latency)
        self._maybe_fail()
        text, usage = self._answer(req, model)
        return Completion(text=text, usage=usage, provider=self.name, model=model)

    async def stream(self, req: ChatRequest, model: str) -> AsyncIterator[StreamChunk]:
        self._maybe_fail()
        text, usage = self._answer(req, model)
        for i in range(0, len(text), 8):
            if self.latency:
                await asyncio.sleep(self.latency / 10)
            yield StreamChunk(text=text[i : i + 8])
        yield StreamChunk(finish_reason="stop", usage=usage)
