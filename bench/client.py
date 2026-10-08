"""Async OpenAI-compatible chat client with per-request timing (TTFT, end-to-end latency).

Every request streams, so time-to-first-token is always measured the same way regardless of
whether the caller consumes tokens incrementally (narrator) or not (factions).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel

Message = dict[str, str]


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int
    temperature: float
    seed: int | None = None
    json_schema: dict[str, Any] | None = None  # set => guided decoding (vLLM structured_outputs)
    priority: int | None = None  # set => requires vLLM --scheduling-policy priority
    return_token_ids: bool = False  # ask vLLM for prompt and sampled token ids (training data)
    # Fixed-length decoding for micro-benchmarks (vLLM sampling extensions); off by default.
    min_tokens: int | None = None
    ignore_eos: bool = False


class Completion(BaseModel):
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_s: float | None
    latency_s: float
    finish_reason: str | None
    error: str | None = None
    prompt_token_ids: list[int] | None = None  # only with ChatRequest.return_token_ids
    token_ids: list[int] | None = None  # sampled ids, as generated (not re-tokenized text)


class ChatClient(Protocol):
    async def chat(
        self, request: ChatRequest, on_token: Callable[[str], None] | None = None
    ) -> Completion: ...


class OpenAIChatClient:
    """Talks to vLLM's OpenAI-compatible server. Retries are disabled: they would distort timing."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout_s,
            max_retries=0,
            http_client=http_client,
        )

    async def chat(
        self, request: ChatRequest, on_token: Callable[[str], None] | None = None
    ) -> Completion:
        extra_body: dict[str, Any] = {}
        if request.json_schema is not None:
            extra_body["structured_outputs"] = {"json": request.json_schema}
        if request.priority is not None:
            extra_body["priority"] = request.priority
        if request.return_token_ids:
            extra_body["return_token_ids"] = True
        if request.min_tokens is not None:
            extra_body["min_tokens"] = request.min_tokens
        if request.ignore_eos:
            extra_body["ignore_eos"] = True

        parts: list[str] = []
        prompt_ids: list[int] | None = None
        ids: list[int] | None = [] if request.return_token_ids else None
        ttft: float | None = None
        usage = None
        finish_reason: str | None = None
        error: str | None = None
        start = time.perf_counter()
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=request.messages,  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                seed=request.seed,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body or None,
            )
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if ids is not None and prompt_ids is None:
                    prompt_ids = getattr(chunk, "prompt_token_ids", None)  # vLLM extension
                for choice in chunk.choices:
                    if ids is not None:
                        ids.extend(getattr(choice, "token_ids", None) or [])
                    delta = choice.delta.content if choice.delta else None
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        parts.append(delta)
                        if on_token is not None:
                            on_token(delta)
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
        except openai.APIError as e:
            error = f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - start
        return Completion(
            text="".join(parts),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            ttft_s=ttft,
            latency_s=latency,
            finish_reason=finish_reason,
            error=error,
            prompt_token_ids=prompt_ids,
            token_ids=ids,
        )

    async def aclose(self) -> None:
        await self._client.close()
