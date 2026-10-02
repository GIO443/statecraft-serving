"""In-process stand-in for the vLLM server, for engine tests without a GPU."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable

from bench.client import ChatRequest, Completion

NARRATION = "The realm held its breath."


def is_narrator(request: ChatRequest) -> bool:
    return "chronicler" in request.messages[-1]["content"]


def faction_of(request: ChatRequest) -> int:
    match = re.search(r"You are faction (\d+)", request.messages[-1]["content"])
    assert match is not None
    return int(match.group(1))


def pass_response(request: ChatRequest) -> str:
    if is_narrator(request):
        return NARRATION
    fid = faction_of(request)
    return json.dumps({"action": {"type": "pass"}, "diplomatic_message": f"faction {fid} waits"})


class FakeClient:
    def __init__(
        self,
        respond: Callable[[ChatRequest], str] = pass_response,
        delay_s: float = 0.0,
        error_for: Callable[[ChatRequest], bool] = lambda _: False,
    ) -> None:
        self.respond = respond
        self.delay_s = delay_s
        self.error_for = error_for
        self.requests: list[ChatRequest] = []

    async def chat(
        self, request: ChatRequest, on_token: Callable[[str], None] | None = None
    ) -> Completion:
        self.requests.append(request)
        await asyncio.sleep(self.delay_s)
        if self.error_for(request):
            return Completion(
                text="",
                prompt_tokens=None,
                completion_tokens=None,
                ttft_s=None,
                latency_s=self.delay_s,
                finish_reason=None,
                error="APIConnectionError: boom",
            )
        text = self.respond(request)
        if on_token is not None:
            for word in text.split(" "):
                on_token(word + " ")
        return Completion(
            text=text,
            prompt_tokens=sum(len(m["content"]) for m in request.messages),
            completion_tokens=len(text),
            ttft_s=self.delay_s / 2,
            latency_s=self.delay_s,
            finish_reason="stop",
        )
