"""Circuit breaker, token-bucket rate limiter and response cache."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

Clock = Callable[[], float]


class CircuitBreaker:
    """Classic three-state breaker per upstream target.

    closed    -> requests flow; `threshold` consecutive failures open it
    open      -> requests are skipped until `cooldown` has passed
    half-open -> exactly one trial request; success closes, failure re-opens
    """

    def __init__(self, threshold: int, cooldown: float, clock: Clock = time.monotonic):
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self.trial_in_flight = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self.trial_in_flight:
            self.trial_in_flight = True
            return True
        return False

    def success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.trial_in_flight = False

    def failure(self) -> None:
        self.trial_in_flight = False
        self.failures += 1
        if self.opened_at is not None or self.failures >= self.threshold:
            self.opened_at = self.clock()  # (re)open; a failed trial restarts the cooldown


class TokenBucket:
    """Refills continuously at `rate_per_min`/60 per second up to `capacity`."""

    def __init__(self, rate_per_min: float, clock: Clock = time.monotonic):
        self.capacity = float(rate_per_min)
        self.rate = rate_per_min / 60.0
        self.tokens = float(rate_per_min)
        self.clock = clock
        self.last = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def try_take(self, n: float) -> float:
        """Take n tokens if available and return 0, else return seconds to wait."""
        self._refill()
        if n <= self.tokens:
            self.tokens -= n
            return 0.0
        if n > self.capacity:
            return float("inf")
        return (n - self.tokens) / self.rate

    def give_back(self, n: float) -> None:
        self._refill()
        self.tokens = min(self.capacity, self.tokens + n)

    def charge(self, n: float) -> None:
        """Record extra usage discovered after the fact; may go negative (debt)."""
        self._refill()
        self.tokens -= n


@dataclass
class KeyLimiter:
    requests: TokenBucket
    tokens: TokenBucket


class TTLCache:
    """LRU cache with a TTL, for exact-match responses."""

    def __init__(self, max_entries: int, ttl: float, clock: Clock = time.monotonic):
        self.max_entries = max_entries
        self.ttl = ttl
        self.clock = clock
        self.data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        item = self.data.get(key)
        if item is None or self.clock() - item[0] > self.ttl:
            if item is not None:
                del self.data[key]
            self.misses += 1
            return None
        self.data.move_to_end(key)
        self.hits += 1
        return item[1]

    def put(self, key: str, value: object) -> None:
        self.data[key] = (self.clock(), value)
        self.data.move_to_end(key)
        while len(self.data) > self.max_entries:
            self.data.popitem(last=False)


def cache_key(alias: str, payload: dict) -> str:
    """Stable hash of the parts of a request that affect the answer."""
    keep = {k: payload.get(k) for k in ("messages", "max_tokens", "max_completion_tokens", "temperature", "stop",
                                        "reasoning_effort")}
    raw = json.dumps({"alias": alias, **keep}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class Spend:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
