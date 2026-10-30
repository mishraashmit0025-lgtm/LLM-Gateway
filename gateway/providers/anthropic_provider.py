"""Anthropic Messages API behind the OpenAI chat format, using the official SDK.

Translation notes:
- system and developer messages become the top-level `system` field.
- `temperature` is not forwarded: current Claude models reject sampling parameters.
- `reasoning_effort` maps to `output_config.effort`.
- stop reasons: end_turn/stop_sequence -> stop, max_tokens -> length, refusal -> content_filter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anthropic

from ..schemas import ChatRequest, Completion, FinishReason, StreamChunk, Usage
from .base import Provider, ProviderError

FINISH: dict[str | None, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def build_params(req: ChatRequest, model: str, default_max_tokens: int) -> dict:
    system = [m.content for m in req.messages if m.role in ("system", "developer")]
    messages = [{"role": m.role, "content": m.content} for m in req.messages if m.role in ("user", "assistant")]
    if not messages:
        raise ProviderError("at least one user message is required", status=400, retryable=False)
    if messages[0]["role"] != "user":
        raise ProviderError("the first non-system message must come from the user", status=400, retryable=False)
    params: dict = {"model": model, "max_tokens": req.output_limit(default_max_tokens), "messages": messages}
    if system:
        params["system"] = "\n\n".join(system)
    if stops := req.stop_list():
        params["stop_sequences"] = stops
    if req.reasoning_effort:
        params["output_config"] = {"effort": req.reasoning_effort}
    return params


def text_of(message) -> str:
    return "".join(b.text for b in message.content if b.type == "text")


def usage_of(message) -> Usage:
    u = message.usage
    return Usage(prompt_tokens=u.input_tokens, completion_tokens=u.output_tokens,
                 total_tokens=u.input_tokens + u.output_tokens)


def translate_error(e: Exception) -> ProviderError:
    if isinstance(e, anthropic.RateLimitError):
        ra = e.response.headers.get("retry-after")
        return ProviderError(f"anthropic rate limited: {e.message}", status=429,
                             retry_after=float(ra) if ra else None)
    if isinstance(e, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return ProviderError(f"anthropic unreachable: {e}", status=504)
    if isinstance(e, anthropic.APIStatusError):
        retryable = e.status_code >= 500 or e.status_code in (408, 409)
        return ProviderError(f"anthropic error {e.status_code}: {e.message}", status=e.status_code,
                             retryable=retryable)
    return ProviderError(f"anthropic: {e}")


class AnthropicProvider(Provider):
    def __init__(self, name: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, default_max_tokens: int = 16000):
        self.name = name
        self.default_max_tokens = default_max_tokens
        # retries and fallback are the gateway's job, so the SDK's own retries are off
        self.client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    async def complete(self, req: ChatRequest, model: str) -> Completion:
        params = build_params(req, model, self.default_max_tokens)
        try:
            msg = await self.client.messages.create(**params)
        except anthropic.AnthropicError as e:
            raise translate_error(e) from e
        return Completion(text=text_of(msg), finish_reason=FINISH.get(msg.stop_reason, "stop"),
                          usage=usage_of(msg), provider=self.name, model=model)

    async def stream(self, req: ChatRequest, model: str) -> AsyncIterator[StreamChunk]:
        params = build_params(req, model, self.default_max_tokens)
        try:
            async with self.client.messages.stream(**params) as s:
                async for event in s:
                    if event.type == "text" and event.text:
                        yield StreamChunk(text=event.text)
                final = await s.get_final_message()
        except anthropic.AnthropicError as e:
            raise translate_error(e) from e
        yield StreamChunk(finish_reason=FINISH.get(final.stop_reason, "stop"), usage=usage_of(final))

    async def aclose(self) -> None:
        await self.client.close()
