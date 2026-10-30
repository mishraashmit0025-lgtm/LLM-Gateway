import json

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.providers.mock import MockProvider

from .conftest import make_config

AUTH = {"Authorization": "Bearer k1"}


def body(text="hello", **kw):
    return {"model": "main", "messages": [{"role": "user", "content": text}], **kw}


@pytest.fixture
def env(clock):
    providers = {n: MockProvider(n) for n in ("a", "b", "c")}
    cfg = make_config(keys={
        "k1": {"name": "alice", "requests_per_minute": 1000, "tokens_per_minute": 1_000_000},
        "k2": {"name": "bob", "requests_per_minute": 2, "tokens_per_minute": 1_000_000},
        "k3": {"name": "carol", "budget_usd": 0.0001},
        "k4": {"name": "dave", "allowed_models": ["lb"]},
    })
    app = create_app(cfg, providers, clock=clock)
    with TestClient(app) as client:
        yield client, providers, clock


def test_auth_required(env):
    client, _, _ = env
    assert client.post("/v1/chat/completions", json=body()).status_code == 401
    assert client.post("/v1/chat/completions", json=body(), headers={"Authorization": "Bearer nope"}).status_code == 401


def test_openai_shaped_response_and_gateway_headers(env):
    client, _, _ = env
    r = client.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion" and j["model"] == "main"
    assert j["choices"][0]["message"] == {"role": "assistant", "content": "[a/big] hello"}
    assert j["choices"][0]["finish_reason"] == "stop"
    assert j["usage"]["total_tokens"] == j["usage"]["prompt_tokens"] + j["usage"]["completion_tokens"]
    assert r.headers["x-gateway-provider"] == "a" and r.headers["x-gateway-model"] == "big"
    assert r.headers["x-gateway-cache"] == "miss"
    assert float(r.headers["x-gateway-cost-usd"]) > 0


def test_validation_and_unknown_model(env):
    client, _, _ = env
    assert client.post("/v1/chat/completions", json={"model": "main", "messages": []}, headers=AUTH).status_code == 400
    r = client.post("/v1/chat/completions", json=body() | {"model": "gpt-9"}, headers=AUTH)
    assert r.status_code == 404 and "unknown model" in r.json()["error"]["message"]


def test_model_allow_list(env):
    client, _, _ = env
    assert client.post("/v1/chat/completions", json=body(), headers={"Authorization": "Bearer k4"}).status_code == 403


def test_cache_hit_skips_the_provider_and_is_free(env):
    client, p, _ = env
    client.post("/v1/chat/completions", json=body("same"), headers=AUTH)
    calls = p["a"].calls
    r = client.post("/v1/chat/completions", json=body("same"), headers=AUTH)
    assert r.headers["x-gateway-cache"] == "hit" and r.headers["x-gateway-cost-usd"] == "0"
    assert p["a"].calls == calls
    # sampling with temperature > 0 is not cached
    r = client.post("/v1/chat/completions", json=body("same", temperature=0.7), headers=AUTH)
    assert r.headers["x-gateway-cache"] == "bypass"


def test_cache_expires(env):
    client, p, clock = env
    client.post("/v1/chat/completions", json=body("ttl"), headers=AUTH)
    clock.advance(301)
    r = client.post("/v1/chat/completions", json=body("ttl"), headers=AUTH)
    assert r.headers["x-gateway-cache"] == "miss"


def test_request_rate_limit_with_retry_after(env):
    client, _, clock = env
    h = {"Authorization": "Bearer k2"}
    assert client.post("/v1/chat/completions", json=body("1"), headers=h).status_code == 200
    assert client.post("/v1/chat/completions", json=body("2"), headers=h).status_code == 200
    r = client.post("/v1/chat/completions", json=body("3"), headers=h)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    clock.advance(31)  # 2/min refills one request every 30s
    assert client.post("/v1/chat/completions", json=body("4"), headers=h).status_code == 200


def test_budget_is_enforced(env):
    client, _, _ = env
    h = {"Authorization": "Bearer k3"}
    big = "x" * 4000  # ~1000 prompt tokens at $4/M = $0.004, over the $0.0001 budget
    assert client.post("/v1/chat/completions", json=body(big), headers=h).status_code == 200
    r = client.post("/v1/chat/completions", json=body("again"), headers=h)
    assert r.status_code == 402
    u = client.get("/v1/usage", headers=h).json()
    assert u["requests"] == 1 and u["cost_usd"] > 0.0001


def test_fallback_is_visible_in_headers_and_metrics(env):
    client, p, _ = env
    p["a"].fail_next = 2
    r = client.post("/v1/chat/completions", json=body("fb"), headers=AUTH)
    assert r.status_code == 200 and r.headers["x-gateway-provider"] == "b"
    m = client.get("/metrics").text
    assert 'gateway_upstream_attempts_total{outcome="error",target="a/big"} 2.0' in m
    assert 'gateway_upstream_attempts_total{outcome="ok",target="b/small"} 1.0' in m


def test_all_upstreams_down_returns_502_with_details(env):
    client, p, _ = env
    p["a"].fail_next = p["b"].fail_next = 10
    r = client.post("/v1/chat/completions", json=body("down"), headers=AUTH)
    assert r.status_code == 502
    assert "a/big" in r.json()["error"]["message"] and "b/small" in r.json()["error"]["message"]


def test_streaming_sse(env):
    client, _, _ = env
    with client.stream("POST", "/v1/chat/completions", json=body("stream me", stream=True), headers=AUTH) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
    assert text == "[a/big] stream me"
    assert events[-1]["choices"][0]["finish_reason"] == "stop" and "usage" in events[-1]
    assert client.get("/v1/usage", headers=AUTH).json()["requests"] == 1


def test_streaming_routing_failure_gets_a_status_code(env):
    client, p, _ = env
    p["a"].fail_next = p["b"].fail_next = 10
    r = client.post("/v1/chat/completions", json=body("x", stream=True), headers=AUTH)
    assert r.status_code == 502


def test_health_reports_circuits(env):
    client, p, _ = env
    p["a"].fail_next = 3
    client.post("/v1/chat/completions", json=body("h1"), headers=AUTH)
    client.post("/v1/chat/completions", json=body("h2"), headers=AUTH)
    assert client.get("/health").json()["circuits"]["a/big"] == "open"
