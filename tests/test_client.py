"""OpenAIChatClient against a mock HTTP transport speaking vLLM's streaming (SSE) format."""

from __future__ import annotations

import asyncio
import json

import httpx

from bench.client import ChatRequest, OpenAIChatClient


def _sse(*events: dict) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def _chunk(content: str | None, finish: str | None = None) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
    }


USAGE_CHUNK = {
    "id": "c1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "m",
    "choices": [],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
}


def _client(handler) -> OpenAIChatClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChatClient("http://test/v1", "m", timeout_s=5, http_client=http)


def test_streaming_completion_and_request_body() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = _sse(_chunk("Hel"), _chunk("lo"), _chunk(None, "stop"), USAGE_CHUNK)
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    tokens: list[str] = []
    schema = {"type": "object"}
    request = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0.5,
        seed=3,
        json_schema=schema,
        priority=-1,
    )
    result = asyncio.run(_client(handler).chat(request, on_token=tokens.append))

    assert result.error is None
    assert result.text == "Hello"
    assert tokens == ["Hel", "lo"]
    assert (result.prompt_tokens, result.completion_tokens) == (11, 3)
    assert result.finish_reason == "stop"
    assert result.ttft_s is not None and 0 <= result.ttft_s <= result.latency_s

    body = seen[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["structured_outputs"] == {"json": schema}
    assert body["priority"] == -1
    assert (body["max_tokens"], body["temperature"], body["seed"]) == (8, 0.5, 3)


def test_optional_fields_omitted() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(_chunk("x", "stop"), USAGE_CHUNK))

    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0)
    asyncio.run(_client(handler).chat(request))
    assert "structured_outputs" not in seen[0]
    assert "priority" not in seen[0]
    assert "min_tokens" not in seen[0] and "ignore_eos" not in seen[0]


def test_fixed_length_fields_sent() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(_chunk("x", "length"), USAGE_CHUNK))

    request = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0,
        min_tokens=8,
        ignore_eos=True,
    )
    asyncio.run(_client(handler).chat(request))
    assert seen[0]["min_tokens"] == 8 and seen[0]["ignore_eos"] is True


def test_server_error_is_reported_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "priority scheduling disabled"}})

    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0)
    result = asyncio.run(_client(handler).chat(request))
    assert result.error is not None and "BadRequestError" in result.error
    assert result.text == ""
    assert result.ttft_s is None


def test_token_ids_requested_and_collected() -> None:
    seen: list[dict] = []

    def chunk(content: str | None, ids: list[int], finish: str | None = None) -> dict:
        c = _chunk(content, finish)
        c["choices"][0]["token_ids"] = ids
        return c

    first = chunk("Hel", [7, 8]) | {"prompt_token_ids": [1, 2, 3]}
    # vLLM sends a content-less chunk for the stop token when token ids are requested.
    events = (first, chunk("lo", [9]), chunk(None, [151645], "stop"), USAGE_CHUNK)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(*events))

    request = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        temperature=0,
        return_token_ids=True,
    )
    result = asyncio.run(_client(handler).chat(request))
    assert seen[0]["return_token_ids"] is True
    assert result.text == "Hello"
    assert result.prompt_token_ids == [1, 2, 3]
    assert result.token_ids == [7, 8, 9, 151645]


def test_token_ids_off_by_default() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(_chunk("x", "stop"), USAGE_CHUNK))

    request = ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0)
    result = asyncio.run(_client(handler).chat(request))
    assert "return_token_ids" not in seen[0]
    assert result.prompt_token_ids is None and result.token_ids is None
