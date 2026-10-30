"""OpenAI-compatible request and response shapes (the subset the gateway serves)."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stop: str | list[str] | None = None
    stream: bool = False
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    user: str | None = None

    def output_limit(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


FinishReason = Literal["stop", "length", "content_filter"]


class Completion(BaseModel):
    """What every provider returns, before it is wrapped as an OpenAI response."""

    text: str
    finish_reason: FinishReason = "stop"
    usage: Usage = Usage()
    provider: str = ""
    model: str = ""


class StreamChunk(BaseModel):
    text: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def to_openai(c: Completion, alias: str, rid: str) -> dict:
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": c.text}, "finish_reason": c.finish_reason}],
        "usage": c.usage.model_dump(),
    }


def chunk_to_openai(ch: StreamChunk, alias: str, rid: str, created: int, first: bool) -> dict:
    delta: dict = {}
    if first:
        delta["role"] = "assistant"
    if ch.text:
        delta["content"] = ch.text
    out = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": alias,
        "choices": [{"index": 0, "delta": delta, "finish_reason": ch.finish_reason}],
    }
    if ch.usage is not None:
        out["usage"] = ch.usage.model_dump()
    return out
