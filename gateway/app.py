"""FastAPI app: an OpenAI-compatible /v1/chat/completions in front of several providers."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import ValidationError

from .config import Config, KeyConfig
from .providers import Provider, ProviderError
from .providers.anthropic_provider import AnthropicProvider
from .providers.mock import MockProvider, rough_tokens
from .providers.openai_provider import OpenAIProvider
from .resilience import KeyLimiter, Spend, TokenBucket, TTLCache, cache_key
from .router import AllTargetsFailed, RouteResult, Router
from .schemas import ChatRequest, Completion, Usage, chunk_to_openai, new_id, to_openai

log = logging.getLogger("gateway")


def build_providers(cfg: Config) -> dict[str, Provider]:
    out: dict[str, Provider] = {}
    for name, p in cfg.providers.items():
        key = os.environ.get(p.api_key_env) if p.api_key_env else None
        if p.type == "anthropic":
            out[name] = AnthropicProvider(name, api_key=key, base_url=p.base_url, timeout=p.timeout_s,
                                          default_max_tokens=cfg.default_max_tokens)
        elif p.type == "openai":
            out[name] = OpenAIProvider(name, api_key=key, base_url=p.base_url, timeout=p.timeout_s)
        else:
            out[name] = MockProvider(name, latency_ms=p.latency_ms, failure_rate=p.failure_rate,
                                     fail_status=p.fail_status)
    return out


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("gateway_requests_total", "Requests by alias and outcome.",
                                ["alias", "outcome"], registry=r)
        self.upstream = Counter("gateway_upstream_attempts_total", "Upstream attempts by target and outcome.",
                                ["target", "outcome"], registry=r)
        self.tokens = Counter("gateway_tokens_total", "Tokens by key, model and direction.",
                              ["key", "model", "direction"], registry=r)
        self.cost = Counter("gateway_cost_usd_total", "Spend in USD by key and model.", ["key", "model"],
                            registry=r)
        self.latency = Histogram("gateway_request_seconds", "End-to-end latency.", ["alias", "stream"],
                                 registry=r, buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120))
        self.cache = Counter("gateway_cache_total", "Cache lookups.", ["result"], registry=r)


class Gateway:
    def __init__(self, cfg: Config, providers: dict[str, Provider], clock: Callable[[], float] = time.monotonic,
                 router: Router | None = None):
        self.cfg = cfg
        self.providers = providers
        self.clock = clock
        self.router = router or Router(cfg, providers, clock=clock)
        self.keys: dict[str, KeyConfig] = cfg.resolved_keys()
        self.limiters: dict[str, KeyLimiter] = {}
        self.spend: dict[str, Spend] = {}
        self.cache = TTLCache(cfg.cache.max_entries, cfg.cache.ttl_s, clock)
        self.metrics = Metrics()

    def limiter(self, key_name: str, kc: KeyConfig) -> KeyLimiter:
        lim = self.limiters.get(key_name)
        if lim is None:
            lim = KeyLimiter(TokenBucket(kc.requests_per_minute, self.clock),
                             TokenBucket(kc.tokens_per_minute, self.clock))
            self.limiters[key_name] = lim
        return lim

    def price(self, model: str, u: Usage) -> float:
        p = self.cfg.pricing.get(model)
        if p is None:
            return 0.0
        return (u.prompt_tokens * p.input_per_mtok + u.completion_tokens * p.output_per_mtok) / 1_000_000

    def record(self, key_name: str, c: Completion) -> float:
        cost = self.price(c.model, c.usage)
        s = self.spend.setdefault(key_name, Spend())
        s.requests += 1
        s.prompt_tokens += c.usage.prompt_tokens
        s.completion_tokens += c.usage.completion_tokens
        s.cost_usd += cost
        s.by_model[c.model] = s.by_model.get(c.model, 0.0) + cost
        m = self.metrics
        m.tokens.labels(key_name, c.model, "prompt").inc(c.usage.prompt_tokens)
        m.tokens.labels(key_name, c.model, "completion").inc(c.usage.completion_tokens)
        m.cost.labels(key_name, c.model).inc(cost)
        return cost


def error(status: int, message: str, kind: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status, headers=headers)


def create_app(cfg: Config, providers: dict[str, Provider] | None = None,
               clock: Callable[[], float] = time.monotonic, router: Router | None = None) -> FastAPI:
    providers = providers if providers is not None else build_providers(cfg)
    gw = Gateway(cfg, providers, clock, router)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for p in providers.values():
            await p.aclose()

    app = FastAPI(title="LLM Gateway", lifespan=lifespan)
    app.state.gw = gw

    def authenticate(request: Request) -> tuple[str, KeyConfig] | JSONResponse:
        if not gw.keys:
            return "anonymous", KeyConfig(name="anonymous", requests_per_minute=10**9, tokens_per_minute=10**12)
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        kc = gw.keys.get(token)
        if kc is None:
            return error(401, "invalid or missing API key", "authentication_error")
        return kc.name, kc

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        started = time.perf_counter()
        auth = authenticate(request)
        if isinstance(auth, JSONResponse):
            return auth
        key_name, kc = auth
        try:
            payload = await request.json()
            req = ChatRequest.model_validate(payload)
        except (ValidationError, ValueError) as e:
            return error(400, f"invalid request: {e}", "invalid_request_error")
        alias = req.model
        if alias not in cfg.routes:
            return error(404, f"unknown model {alias!r}; see GET /v1/models", "invalid_request_error")
        if kc.allowed_models is not None and alias not in kc.allowed_models:
            return error(403, f"key {kc.name!r} may not use {alias!r}", "permission_error")
        spend = gw.spend.get(key_name)
        if kc.budget_usd is not None and spend is not None and spend.cost_usd >= kc.budget_usd:
            return error(402, f"budget of ${kc.budget_usd:.2f} exhausted", "budget_exceeded")

        # rate limits: one request, plus an up-front token estimate reconciled after the call
        lim = gw.limiter(key_name, kc)
        wait = lim.requests.try_take(1)
        if wait:
            gw.metrics.requests.labels(alias, "rate_limited").inc()
            return error(429, "request rate limit exceeded", "rate_limit_error",
                         {"retry-after": str(max(1, int(wait + 0.999)))})
        estimate = sum(rough_tokens(m.content) for m in req.messages) + min(req.output_limit(cfg.default_max_tokens), 4096)
        wait = lim.tokens.try_take(estimate)
        if wait:
            lim.requests.give_back(1)
            gw.metrics.requests.labels(alias, "rate_limited").inc()
            retry = "60" if wait == float("inf") else str(max(1, int(wait + 0.999)))
            return error(429, "token rate limit exceeded", "rate_limit_error", {"retry-after": retry})

        rid = new_id()
        cacheable = cfg.cache.enabled and not req.stream and not req.temperature
        ckey = cache_key(alias, payload) if cacheable else ""
        if cacheable:
            hit = gw.cache.get(ckey)
            gw.metrics.cache.labels("hit" if hit else "miss").inc()
            if hit is not None:
                lim.tokens.give_back(estimate)
                gw.metrics.requests.labels(alias, "cache_hit").inc()
                return JSONResponse(to_openai(hit, alias, rid), headers={
                    "x-request-id": rid, "x-gateway-cache": "hit", "x-gateway-provider": hit.provider,
                    "x-gateway-model": hit.model, "x-gateway-cost-usd": "0"})

        res = RouteResult()
        if req.stream:
            return await stream_response(req, alias, rid, key_name, lim, estimate, res, started)
        try:
            c = await gw.router.complete(alias, req, res)
        except ProviderError as e:
            lim.tokens.give_back(estimate)
            gw.metrics.requests.labels(alias, "client_error").inc()
            return error(e.status, str(e), "invalid_request_error")
        except AllTargetsFailed as e:
            lim.tokens.give_back(estimate)
            observe_attempts(res)
            gw.metrics.requests.labels(alias, "upstream_error").inc()
            return error(e.status, str(e), "upstream_error", {"retry-after": "5"} if e.status in (429, 503) else None)
        observe_attempts(res)
        lim.tokens.charge(c.usage.total_tokens - estimate)
        cost = gw.record(key_name, c)
        if cacheable and c.finish_reason == "stop":
            gw.cache.put(ckey, c)
        gw.metrics.requests.labels(alias, "ok").inc()
        gw.metrics.latency.labels(alias, "false").observe(time.perf_counter() - started)
        log.info("request %s key=%s alias=%s served_by=%s/%s tokens=%d cost=%.6f attempts=%d", rid, key_name, alias,
                 c.provider, c.model, c.usage.total_tokens, cost, len(res.attempts))
        return JSONResponse(to_openai(c, alias, rid), headers={
            "x-request-id": rid, "x-gateway-cache": "miss" if cacheable else "bypass",
            "x-gateway-provider": c.provider, "x-gateway-model": c.model, "x-gateway-cost-usd": f"{cost:.8f}"})

    def observe_attempts(res: RouteResult) -> None:
        for a in res.attempts:
            gw.metrics.upstream.labels(a.target, a.outcome).inc()

    async def stream_response(req: ChatRequest, alias: str, rid: str, key_name: str, lim, estimate: int,
                              res: RouteResult, started: float):
        gen = gw.router.stream(alias, req, res)
        # open the stream before replying, so routing failures still get a proper status code
        try:
            first = await gen.__anext__()
        except ProviderError as e:
            lim.tokens.give_back(estimate)
            return error(e.status, str(e), "invalid_request_error")
        except AllTargetsFailed as e:
            lim.tokens.give_back(estimate)
            observe_attempts(res)
            gw.metrics.requests.labels(alias, "upstream_error").inc()
            return error(e.status, str(e), "upstream_error")
        observe_attempts(res)
        target = res.target
        created = int(time.time())

        async def body():
            text_parts: list[str] = []
            usage = Usage()
            finish = "stop"
            is_first = True
            chunk = first
            try:
                while True:
                    if chunk.text:
                        text_parts.append(chunk.text)
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.finish_reason:
                        finish = chunk.finish_reason
                    yield f"data: {json.dumps(chunk_to_openai(chunk, alias, rid, created, is_first))}\n\n"
                    is_first = False
                    chunk = await gen.__anext__()
            except StopAsyncIteration:
                pass
            except ProviderError as e:
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_error'}})}\n\n"
                gw.metrics.requests.labels(alias, "stream_interrupted").inc()
            c = Completion(text="".join(text_parts), finish_reason=finish, usage=usage,  # type: ignore[arg-type]
                           provider=target.provider, model=target.model)
            lim.tokens.charge(usage.total_tokens - estimate)
            gw.record(key_name, c)
            gw.metrics.requests.labels(alias, "ok").inc()
            gw.metrics.latency.labels(alias, "true").observe(time.perf_counter() - started)
            yield "data: [DONE]\n\n"

        return StreamingResponse(body(), media_type="text/event-stream", headers={
            "x-request-id": rid, "x-gateway-provider": target.provider, "x-gateway-model": target.model,
            "cache-control": "no-cache"})

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": a, "object": "model", "owned_by": "gateway"} for a in cfg.routes]}

    @app.get("/v1/usage")
    async def usage(request: Request):
        auth = authenticate(request)
        if isinstance(auth, JSONResponse):
            return auth
        key_name, kc = auth
        s = gw.spend.get(key_name, Spend())
        return {"key": key_name, "requests": s.requests, "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens, "cost_usd": round(s.cost_usd, 8),
                "budget_usd": kc.budget_usd, "cost_by_model": s.by_model}

    @app.get("/health")
    async def health():
        return {"status": "ok", "circuits": {k: b.state for k, b in gw.router.breakers.items()},
                "cache": {"entries": len(gw.cache.data), "hits": gw.cache.hits, "misses": gw.cache.misses}}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(generate_latest(gw.metrics.registry).decode(),
                                 media_type="text/plain; version=0.0.4")

    return app
