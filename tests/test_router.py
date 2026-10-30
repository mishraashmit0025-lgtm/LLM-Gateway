import random

import pytest

from gateway.providers import ProviderError
from gateway.providers.mock import MockProvider
from gateway.router import AllTargetsFailed, RouteResult, Router
from gateway.schemas import ChatRequest

from .conftest import make_config

REQ = ChatRequest(model="main", messages=[{"role": "user", "content": "hello"}])


def setup(clock, **cfg):
    providers = {n: MockProvider(n) for n in ("a", "b", "c")}
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    router = Router(make_config(**cfg), providers, rng=random.Random(0), clock=clock, sleep=fake_sleep)
    return router, providers, sleeps


async def test_primary_success(clock):
    router, p, _ = setup(clock)
    res = RouteResult()
    c = await router.complete("main", REQ, res)
    assert c.provider == "a" and c.model == "big"
    assert [a.outcome for a in res.attempts] == ["ok"]


async def test_retry_then_success_on_same_target(clock):
    router, p, sleeps = setup(clock)
    p["a"].fail_next = 1
    res = RouteResult()
    c = await router.complete("main", REQ, res)
    assert c.provider == "a"
    assert [a.outcome for a in res.attempts] == ["error", "ok"]
    assert len(sleeps) == 1


async def test_falls_back_after_retries_are_exhausted(clock):
    router, p, _ = setup(clock)
    p["a"].fail_next = 2
    res = RouteResult()
    c = await router.complete("main", REQ, res)
    assert c.provider == "b" and c.model == "small"
    assert [a.target for a in res.attempts] == ["a/big", "a/big", "b/small"]


async def test_client_errors_are_not_retried_or_failed_over(clock):
    router, p, _ = setup(clock)
    p["a"].fail_status = 400
    p["a"].fail_next = 1
    with pytest.raises(ProviderError) as e:
        await router.complete("main", REQ, RouteResult())
    assert e.value.status == 400
    assert p["b"].calls == 0
    assert router.breakers["a/big"].state == "closed"  # a bad request is not the provider's fault


async def test_open_circuit_is_skipped_until_cooldown(clock):
    router, p, _ = setup(clock)
    p["a"].fail_next = 3  # threshold is 3
    await router.complete("main", REQ, RouteResult())  # a fails twice, b serves
    await router.complete("main", REQ, RouteResult())  # a fails once more -> open
    assert router.breakers["a/big"].state == "open"
    calls = p["a"].calls
    res = RouteResult()
    c = await router.complete("main", REQ, res)
    assert c.provider == "b" and p["a"].calls == calls
    assert res.attempts[0].outcome == "skipped (circuit open)"
    clock.advance(30)
    c = await router.complete("main", REQ, RouteResult())  # half-open trial succeeds
    assert c.provider == "a" and router.breakers["a/big"].state == "closed"


async def test_all_rate_limited_reports_429(clock):
    router, p, _ = setup(clock)
    for n in ("a", "b"):
        p[n].fail_status = 429
        p[n].fail_next = 10
    with pytest.raises(AllTargetsFailed) as e:
        await router.complete("main", REQ, RouteResult())
    assert e.value.status == 429


async def test_retry_after_header_is_respected(clock):
    router, p, sleeps = setup(clock)

    async def flaky(req, model):
        raise ProviderError("slow down", status=429, retry_after=0.03)

    p["a"].complete = flaky
    await router.complete("main", REQ, RouteResult())
    assert sleeps == [0.03]


async def test_stream_falls_back_before_first_chunk(clock):
    router, p, _ = setup(clock)
    p["a"].fail_next = 2
    res = RouteResult()
    text = "".join([ch.text async for ch in router.stream("main", REQ, res)])
    assert text == "[b/small] hello"
    assert res.target.provider == "b"
