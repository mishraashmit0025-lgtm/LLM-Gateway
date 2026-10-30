import random
from collections import Counter

from gateway.config import Target
from gateway.resilience import CircuitBreaker, TokenBucket, TTLCache, cache_key
from gateway.router import weighted_order


def test_breaker_opens_half_opens_and_closes(clock):
    b = CircuitBreaker(threshold=3, cooldown=10, clock=clock)
    for _ in range(2):
        b.failure()
    assert b.state == "closed" and b.allow()
    b.failure()
    assert b.state == "open" and not b.allow()
    clock.advance(10)
    assert b.state == "half_open"
    assert b.allow()  # the single trial request
    assert not b.allow()  # nobody else while it is in flight
    b.success()
    assert b.state == "closed" and b.failures == 0


def test_failed_trial_reopens_for_a_full_cooldown(clock):
    b = CircuitBreaker(threshold=1, cooldown=10, clock=clock)
    b.failure()
    clock.advance(10)
    assert b.allow()
    b.failure()
    assert b.state == "open"
    clock.advance(9)
    assert b.state == "open"
    clock.advance(1)
    assert b.state == "half_open"


def test_token_bucket_refills_continuously(clock):
    tb = TokenBucket(60, clock)  # one per second
    for _ in range(60):
        assert tb.try_take(1) == 0
    wait = tb.try_take(1)
    assert 0.99 < wait <= 1.0
    clock.advance(1.5)
    assert tb.try_take(1) == 0
    assert tb.try_take(1000) == float("inf")  # can never fit


def test_token_bucket_debt_delays_future_requests(clock):
    tb = TokenBucket(600, clock)  # 10 per second
    assert tb.try_take(100) == 0
    tb.charge(600)  # the call turned out much bigger than estimated
    assert tb.try_take(1) > 0
    clock.advance(20)
    assert tb.try_take(1) == 0


def test_cache_ttl_and_lru(clock):
    c = TTLCache(max_entries=2, ttl=10, clock=clock)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # evicts b, the least recently used
    assert c.get("b") is None and c.get("a") == 1
    clock.advance(11)
    assert c.get("a") is None


def test_cache_key_ignores_irrelevant_fields_and_key_order():
    p1 = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "user": "alice", "stream": False}
    p2 = {"stream": False, "messages": [{"content": "hi", "role": "user"}], "model": "x", "user": "bob"}
    assert cache_key("x", p1) == cache_key("x", p2)
    assert cache_key("x", p1) != cache_key("y", p1)
    assert cache_key("x", p1) != cache_key("x", {**p1, "max_tokens": 5})


def test_weighted_order_matches_weights():
    rng = random.Random(1)
    ts = [Target(provider="a", model="m", weight=3), Target(provider="b", model="m", weight=1)]
    first = Counter(weighted_order(ts, rng)[0].provider for _ in range(20000))
    share = first["a"] / 20000
    assert 0.73 < share < 0.77  # expected 0.75
