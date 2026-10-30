"""Runs the real Anthropic SDK against a local fake of the Messages API, so the
translation is checked on the wire (headers, JSON body, SSE events) and not
just against our own assumptions."""

import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.providers import ProviderError
from gateway.providers.anthropic_provider import AnthropicProvider
from gateway.schemas import ChatRequest

seen: list[dict] = []
fake = FastAPI()


def message(text: str, stop_reason: str = "end_turn") -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 5}}


@fake.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    seen.append({"body": body, "headers": dict(request.headers)})
    first = body["messages"][0]["content"]
    if first == "overloaded":
        return JSONResponse({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}, status_code=529)
    if first == "bad":
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "nope"}},
                            status_code=400)
    if first == "limit":
        return JSONResponse({"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}},
                            status_code=429, headers={"retry-after": "7"})
    if not body.get("stream"):
        return JSONResponse(message("Hello there", "max_tokens" if first == "long" else "end_turn"))

    def sse():
        def ev(name, data):
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"
        start = message("")
        start["content"], start["stop_reason"] = [], None
        start["usage"] = {"input_tokens": 12, "output_tokens": 1}
        yield ev("message_start", {"type": "message_start", "message": start})
        yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                         "content_block": {"type": "text", "text": ""}})
        for piece in ["Hel", "lo ", "there"]:
            yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                             "delta": {"type": "text_delta", "text": piece}})
        yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                   "usage": {"output_tokens": 5}})
        yield ev("message_stop", {"type": "message_stop"})

    return StreamingResponse(sse(), media_type="text/event-stream")


@pytest.fixture(scope="module")
def base_url():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(5)


def provider(base_url):
    return AnthropicProvider("anthropic", api_key="test-key", base_url=base_url, timeout=10)


def req(text, **kw):
    return ChatRequest(model="smart", messages=[
        {"role": "system", "content": "Be brief."},
        {"role": "developer", "content": "Answer in English."},
        {"role": "user", "content": text},
    ], **kw)


async def test_request_translation_and_response(base_url):
    seen.clear()
    c = await provider(base_url).complete(req("hi", max_tokens=50, temperature=0.5, stop="END", reasoning_effort="low"),
                                          "claude-opus-5-5")
    assert c.text == "Hello there" and c.finish_reason == "stop"
    assert (c.usage.prompt_tokens, c.usage.completion_tokens, c.usage.total_tokens) == (12, 5, 17)
    sent = seen[-1]
    b = sent["body"]
    assert b["model"] == "claude-opus-5-5" and b["max_tokens"] == 50
    assert b["system"] == "Be brief.\n\nAnswer in English."
    assert b["messages"] == [{"role": "user", "content": "hi"}]
    assert b["stop_sequences"] == ["END"]
    assert b["output_config"] == {"effort": "low"}
    assert "temperature" not in b  # current Claude models reject sampling parameters
    h = sent["headers"]
    assert h.get("x-api-key") == "test-key" or h.get("authorization") == "Bearer test-key"
    assert "anthropic-version" in sent["headers"]


async def test_max_tokens_maps_to_length(base_url):
    c = await provider(base_url).complete(req("long"), "claude-opus-5-5")
    assert c.finish_reason == "length"


async def test_streaming(base_url):
    chunks = [ch async for ch in provider(base_url).stream(req("hi"), "claude-opus-5-5")]
    assert "".join(c.text for c in chunks) == "Hello there"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage.completion_tokens == 5 and chunks[-1].usage.prompt_tokens == 12


async def test_error_classification(base_url):
    p = provider(base_url)
    with pytest.raises(ProviderError) as e:
        await p.complete(req("overloaded"), "claude-opus-5-5")
    assert e.value.status == 529 and e.value.retryable
    with pytest.raises(ProviderError) as e:
        await p.complete(req("bad"), "claude-opus-5-5")
    assert e.value.status == 400 and not e.value.retryable
    with pytest.raises(ProviderError) as e:
        await p.complete(req("limit"), "claude-opus-5-5")
    assert e.value.status == 429 and e.value.retryable and e.value.retry_after == 7


async def test_conversation_must_start_with_user():
    p = AnthropicProvider("anthropic", api_key="x", base_url="http://127.0.0.1:9")
    bad = ChatRequest(model="smart", messages=[{"role": "assistant", "content": "hi"}])
    with pytest.raises(ProviderError) as e:
        await p.complete(bad, "claude-opus-5-5")
    assert e.value.status == 400 and not e.value.retryable
