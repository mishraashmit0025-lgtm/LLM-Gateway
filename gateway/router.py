"""Picks upstream targets for a model alias and drives retries and fallback."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from .config import Config, Target
from .providers import Provider, ProviderError
from .resilience import CircuitBreaker
from .schemas import ChatRequest, Completion, StreamChunk


@dataclass
class Attempt:
    target: str
    outcome: str  # "ok", "error", "skipped (circuit open)"
    status: int | None = None
    detail: str = ""


class AllTargetsFailed(Exception):
    def __init__(self, alias: str, attempts: list[Attempt]):
        self.attempts = attempts
        statuses = {a.status for a in attempts if a.status}
        # if every upstream was rate limiting us, tell the client that, not 502
        self.status = 429 if statuses == {429} else 503 if not statuses else 502
        summary = "; ".join(f"{a.target}: {a.outcome} {a.status or ''} {a.detail}".strip() for a in attempts)
        super().__init__(f"all targets for {alias!r} failed: {summary}")


@dataclass
class RouteResult:
    target: Target | None = None
    attempts: list[Attempt] = field(default_factory=list)


def weighted_order(targets: list[Target], rng: random.Random) -> list[Target]:
    """Weighted shuffle without replacement (Efraimidis-Spirakis): each target
    gets key u^(1/w); sorting by key gives an order where a target appears
    first with probability proportional to its weight."""
    keyed = [(rng.random() ** (1.0 / t.weight) if t.weight > 0 else 0.0, t) for t in targets]
    return [t for _, t in sorted(keyed, key=lambda kt: kt[0], reverse=True)]


class Router:
    def __init__(self, cfg: Config, providers: dict[str, Provider], *, rng: random.Random | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.cfg = cfg
        self.providers = providers
        self.rng = rng or random.Random()
        self.clock = clock
        self.sleep = sleep
        self.breakers: dict[str, CircuitBreaker] = {}

    def breaker(self, t: Target) -> CircuitBreaker:
        key = f"{t.provider}/{t.model}"
        b = self.breakers.get(key)
        if b is None:
            b = CircuitBreaker(self.cfg.breaker.failure_threshold, self.cfg.breaker.cooldown_s, self.clock)
            self.breakers[key] = b
        return b

    def plan(self, alias: str) -> list[Target]:
        route = self.cfg.routes[alias]
        return weighted_order(route.primary, self.rng) + list(route.fallbacks)

    def backoff(self, attempt: int, err: ProviderError) -> float:
        r = self.cfg.retry
        if err.retry_after is not None:
            return min(err.retry_after, r.max_delay_s)
        return self.rng.uniform(0, min(r.max_delay_s, r.base_delay_s * 2**attempt))

    async def _run(self, alias: str, res: RouteResult, call: Callable[[Target], Awaitable[object]]):
        """Try targets in plan order with retries; return the first success."""
        for t in self.plan(alias):
            name = f"{t.provider}/{t.model}"
            br = self.breaker(t)
            if not br.allow():
                res.attempts.append(Attempt(name, "skipped (circuit open)"))
                continue
            for attempt in range(self.cfg.retry.attempts):
                try:
                    out = await call(t)
                except ProviderError as e:
                    if not e.retryable:
                        # the request itself is bad; another provider will not fix it
                        br.success()
                        raise
                    br.failure()
                    res.attempts.append(Attempt(name, "error", e.status, str(e)))
                    if attempt + 1 < self.cfg.retry.attempts and br.state == "closed":
                        await self.sleep(self.backoff(attempt, e))
                        continue
                    break
                br.success()
                res.attempts.append(Attempt(name, "ok", 200))
                res.target = t
                return out
        raise AllTargetsFailed(alias, res.attempts)

    async def complete(self, alias: str, req: ChatRequest, res: RouteResult) -> Completion:
        async def call(t: Target) -> Completion:
            return await self.providers[t.provider].complete(req, t.model)

        return await self._run(alias, res, call)  # type: ignore[return-value]

    async def stream(self, alias: str, req: ChatRequest, res: RouteResult) -> AsyncIterator[StreamChunk]:
        """Fallback is only possible before the first chunk reaches the client;
        after that an upstream error ends the stream."""

        async def open_stream(t: Target):
            agen = self.providers[t.provider].stream(req, t.model)
            try:
                first = await agen.__anext__()
            except StopAsyncIteration:
                first = StreamChunk(finish_reason="stop")
            return first, agen

        first, agen = await self._run(alias, res, open_stream)  # type: ignore[misc]
        yield first
        async for chunk in agen:
            yield chunk
