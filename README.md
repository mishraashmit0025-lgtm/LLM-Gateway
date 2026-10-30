# LLM Gateway

One OpenAI-compatible endpoint in front of several LLM providers. Applications send `/v1/chat/completions` requests to the gateway, using any OpenAI client library. The gateway decides which provider and model actually serves each request, retries and fails over when a provider has trouble, enforces per-key rate limits and budgets, caches repeated requests, and records what everything costs.

Supported upstreams: Anthropic (through the official `anthropic` SDK), OpenAI, and any OpenAI-compatible server such as vLLM, Ollama or LM Studio. A mock provider is built in for demos and tests.

## Why

Teams that ship LLM features run into the same operational problems: one provider has an outage or rate limits you, the bill grows with no attribution, and every service reimplements retries. Putting that logic in one gateway keeps application code simple and makes the policy visible in one config file.

## Features

- **Model aliases and routing.** Clients ask for `smart` or `fast`. The config maps each alias to weighted primary targets (load-balanced with a weighted shuffle) and an ordered list of fallbacks.
- **Retries with jittered backoff** on retryable errors (429, 5xx, timeouts). `Retry-After` from the upstream is respected. Client errors such as a malformed request are returned immediately, not retried on another provider.
- **Circuit breaker per upstream target** (closed, open, half-open with a single trial request), so a dead provider is skipped instead of adding latency to every request.
- **Streaming (SSE)** that falls back to another provider if the first one fails before sending a token. After the first token, an upstream error ends the stream with an error event.
- **Per-key token-bucket rate limits** on requests and tokens. Tokens are estimated up front and reconciled with the real usage afterwards, so a request that turned out larger than estimated is paid back from future capacity. Rejected requests get `429` with `Retry-After`.
- **Per-key budgets** (402 once spent) and per-key model allow-lists.
- **Exact-match response cache** with TTL and LRU eviction, used only for deterministic requests (no temperature), and keyed on a canonical hash of the fields that affect the answer.
- **Cost accounting** per key and model from a pricing table, exposed at `/v1/usage` and as Prometheus counters. Every response carries `x-gateway-provider`, `x-gateway-model`, `x-gateway-cache` and `x-gateway-cost-usd` headers.
- **Provider translation.** For Anthropic: system and developer messages become the `system` field, `stop` becomes `stop_sequences`, `reasoning_effort` becomes `output_config.effort`, stop reasons map to OpenAI finish reasons, and `temperature` is dropped because current Claude models reject sampling parameters.
- **Observability.** Prometheus metrics at `/metrics` (requests by outcome, upstream attempts by target and outcome, tokens, cost, latency histogram, cache hits). Circuit states and cache stats are at `/health`.

## Quick start

```bash
pip install -e ".[dev]"

# no API keys needed: two mock providers, one of which fails 30% of the time
python -m gateway --config config.demo.yaml --port 8000

curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-demo" -H "Content-Type: application/json" \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "hello"}]}' -i
```

With real providers, copy `config.example.yaml`, set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`, and point any OpenAI client at the gateway:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-gw-dev-local")
r = client.chat.completions.create(model="smart", messages=[{"role": "user", "content": "Explain Raft in two lines"}])
print(r.choices[0].message.content)
```

Docker:

```bash
docker build -t llm-gateway .
docker run -p 8000:8000 -v $PWD/config.example.yaml:/app/config.yaml -e ANTHROPIC_API_KEY llm-gateway
```

## Configuration

```yaml
routes:
  smart:
    primary:
      - {provider: anthropic, model: claude-opus-5-5}
    fallbacks:
      - {provider: anthropic, model: claude-sonnet-5-5}
      - {provider: openai, model: gpt-5.4}
  fast:
    primary:                     # 75% / 25% split
      - {provider: anthropic, model: claude-haiku-5-5, weight: 3}
      - {provider: local, model: llama3.1:8b, weight: 1}

keys:
  sk-gw-dev-local: {name: dev, requests_per_minute: 60, tokens_per_minute: 200000, budget_usd: 5.00}

retry:   {attempts: 2, base_delay_s: 0.25, max_delay_s: 4}
breaker: {failure_threshold: 5, cooldown_s: 30}
cache:   {enabled: true, ttl_s: 300, max_entries: 10000}
```

Provider API keys are only ever read from environment variables. Gateway keys can also come from an environment variable (`keys_env`) so that no secret lives in the file. See `config.example.yaml` for everything.

## Load test

`scripts/loadtest.py` starts the gateway in its own process with mock upstreams and sends 3,000 requests from 50 concurrent connections (Windows 11 laptop, one uvicorn worker):

| Scenario | Throughput | Latency | Success |
|---|---|---|---|
| Zero-latency upstream (pure gateway overhead) | 1,080 req/s | p50 46 ms, p99 63 ms at 50 in flight; about 2.7 ms per request when sequential | 100% |
| Primary fails 30% of the time | 1,130 req/s | p50 40 ms, p99 73 ms | 100% (303 requests were served by the fallback) |

The p50 under load is mostly queueing (50 requests in flight on one worker). Run more uvicorn workers to scale.

## Tests

```bash
pytest -q
```

33 tests. The router tests use a fake clock and a fake `sleep`, so retries, backoff and circuit cooldowns are deterministic. The app tests drive the real FastAPI app: auth, validation, allow-lists, cache hits and expiry, request and token rate limits with `Retry-After`, budgets, failover visible in headers and metrics, all-upstreams-down errors, SSE streaming, and stream failover. `test_anthropic_wire.py` runs the real Anthropic SDK against a local fake of the Messages API. That checks the translation on the wire: the request body, auth headers, SSE events, usage, stop reasons, and how 400 / 429 (with `retry-after`) / 529 overloaded errors are classified.

## Layout

```
gateway/app.py                 HTTP API, auth, limits, budgets, cache, SSE
gateway/router.py              target selection, retries, failover, streaming failover
gateway/resilience.py          circuit breaker, token bucket, TTL/LRU cache
gateway/config.py              YAML config model
gateway/schemas.py             OpenAI-compatible request/response shapes
gateway/providers/             anthropic (official SDK), openai, mock
scripts/loadtest.py            throughput and chaos test
```

## Limits

State (rate limit buckets, spend, cache) lives in process memory, so it is per gateway instance. Running several replicas behind a load balancer would need that moved to Redis. Tool calling and image inputs are not translated yet; requests are text chat only.
