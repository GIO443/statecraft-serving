"""Faction agents and narrator: turn prompts into requests and model output into actions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from bench.client import ChatClient, ChatRequest, Completion, Message
from sim.actions import ActionResponse, parse_response, response_json_schema
from sim.world import GameConfig

NarratorMode = Literal["off", "sequential", "pipelined"]


class AgentConfig(BaseModel):
    """How agents call the model. Loaded from the experiment config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    faction_max_tokens: int
    narrator_max_tokens: int
    temperature: float
    guided_decoding: bool
    faction_priority: int | None  # None => field not sent (server not in priority mode)
    narrator_priority: int | None
    # off: no narrator and no history; sequential: narrate before the next turn starts;
    # pipelined: narrate turn t concurrently with turn t+1's decisions (history lags one turn).
    narrator: NarratorMode


class Decision(BaseModel):
    faction: int
    completion: Completion
    response: ActionResponse | None
    parse_error: str | None


def request_seed(game_seed: int, turn: int, actor: str) -> int:
    """Stable per-request sampling seed (sampling param only; never enters the prompt)."""
    digest = hashlib.sha256(f"{game_seed}:{turn}:{actor}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


class Agents:
    def __init__(self, client: ChatClient, game_cfg: GameConfig, agent_cfg: AgentConfig) -> None:
        self.client = client
        self.game_cfg = game_cfg
        self.agent_cfg = agent_cfg
        self._schema: dict[str, Any] | None = (
            response_json_schema(game_cfg.max_message_chars) if agent_cfg.guided_decoding else None
        )

    async def decide(self, fid: int, messages: list[Message], seed: int) -> Decision:
        request = ChatRequest(
            messages=messages,
            max_tokens=self.agent_cfg.faction_max_tokens,
            temperature=self.agent_cfg.temperature,
            seed=seed,
            json_schema=self._schema,
            priority=self.agent_cfg.faction_priority,
        )
        completion = await self.client.chat(request)
        if completion.error is not None:
            return Decision(
                faction=fid,
                completion=completion,
                response=None,
                parse_error=f"request failed: {completion.error}",
            )
        parsed = parse_response(completion.text, self.game_cfg.max_message_chars)
        return Decision(
            faction=fid, completion=completion, response=parsed.response, parse_error=parsed.error
        )

    async def narrate(
        self, messages: list[Message], seed: int, on_token: Callable[[str], None] | None
    ) -> Completion:
        request = ChatRequest(
            messages=messages,
            max_tokens=self.agent_cfg.narrator_max_tokens,
            temperature=self.agent_cfg.temperature,
            seed=seed,
            priority=self.agent_cfg.narrator_priority,
        )
        return await self.client.chat(request, on_token=on_token)
