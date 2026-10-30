"""OpenAI (or any OpenAI-compatible server such as vLLM or Ollama) as an upstream."""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai

from ..schemas import ChatRequest, Completion, FinishReason, StreamChunk, Usage
from .base import Provider, ProviderError


def _finish(reason: str | None) -> FinishReason:
    if reason == "length":
        return "length"
    if reason == "content_filter":
        return "content_filter"
    return "stop"


def translate_error(e: Exception) -> ProviderError:
    if isinstance(e, openai.RateLimitError):
        ra = e.response.headers.get("retry-after")
        return ProviderError(f"openai rate limited: {e.message}", status=429, retry_after=float(ra) if ra else None)
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return ProviderError(f"openai unreachable: {e}", status=504)
    if isinstance(e, openai.APIStatusError):
        retryable = e.status_code >= 500 or e.status_code in (408, 409)
        return ProviderError(f"openai error {e.status_code}: {e.message}", status=e.status_code, retryable=retryable)
    return ProviderError(f"openai: {e}")


class OpenAIProvider(Provider):
    def __init__(self, name: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0):
        self.name = name
        self.client = openai.AsyncOpenAI(api_key=api_key or "unused", base_url=base_url, timeout=timeout,
                                         max_retries=0)

    def _params(self, req: ChatRequest, model: str) -> dict:
        p: dict = {"model": model, "messages": [m.model_dump() for m in req.messages]}
        if (n := req.max_completion_tokens or req.max_tokens) is not None:
            p["max_completion_tokens"] = n
        if req.temperature is not None:
            p["temperature"] = req.temperature
        if req.stop is not None:
            p["stop"] = req.stop
        if req.reasoning_effort in ("low", "medium", "high"):
            p["reasoning_effort"] = req.reasoning_effort
        return p

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        try:
            r = await self.client.chat.completions.create(**self._params(req, model))
        except openai.OpenAIError as e:
            raise translate_error(e) from e
        u = r.usage
        usage = (Usage(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                       total_tokens=u.total_tokens) if u else Usage())
        ch = r.choices[0]
        return Completion(text=ch.message.content or "", finish_reason=_finish(ch.finish_reason), usage=usage,
                          provider=self.name, model=model)

    async def stream(self, req: ChatRequest, model: str) -> AsyncIterator[StreamChunk]:
        params = self._params(req, model) | {"stream": True, "stream_options": {"include_usage": True}}
        finish: FinishReason = "stop"
        usage = Usage()
        try:
            stream = await self.client.chat.completions.create(**params)
            async for ev in stream:
                if ev.usage:
                    usage = Usage(prompt_tokens=ev.usage.prompt_tokens, completion_tokens=ev.usage.completion_tokens,
                                  total_tokens=ev.usage.total_tokens)
                for ch in ev.choices:
                    if ch.delta and ch.delta.content:
                        yield StreamChunk(text=ch.delta.content)
                    if ch.finish_reason:
                        finish = _finish(ch.finish_reason)
        except openai.OpenAIError as e:
            raise translate_error(e) from e
        yield StreamChunk(finish_reason=finish, usage=usage)

    async def aclose(self) -> None:
        await self.client.close()
