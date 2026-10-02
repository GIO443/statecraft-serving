"""Turn loop: parallel faction decisions -> deterministic resolution -> narrator."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from bench.client import ChatClient, Completion
from sim.agents import AgentConfig, Agents, Decision, request_seed
from sim.prompts import faction_messages, narrator_messages, shared_prefix
from sim.rules import TurnResult, resolve_turn
from sim.world import GameConfig, HistoryEntry, World, generate_world


class RequestRecord(BaseModel):
    """One row of requests.jsonl."""

    turn: int
    actor: Literal["faction", "narrator"]
    faction: int | None
    prompt_tokens: int | None
    output_tokens: int | None
    ttft_s: float | None
    latency_s: float
    finish_reason: str | None
    valid_json: bool | None  # None for narrator
    legal: bool | None  # None for narrator
    error: str | None


class TurnRecord(BaseModel):
    turn: int
    n_factions: int  # living factions that acted this turn
    wall_time_s: float  # turn start -> ready for next turn (includes narrator unless pipelined)
    decide_time_s: float  # all faction requests in flight -> all returned
    requests: list[RequestRecord] = Field(default_factory=list)
    result: TurnResult
    outputs: dict[int, str]  # raw faction completions (training data for the draft head)
    narration: str | None = None


def _faction_record(turn: int, d: Decision, result: TurnResult) -> RequestRecord:
    c = d.completion
    return RequestRecord(
        turn=turn,
        actor="faction",
        faction=d.faction,
        prompt_tokens=c.prompt_tokens,
        output_tokens=c.completion_tokens,
        ttft_s=c.ttft_s,
        latency_s=c.latency_s,
        finish_reason=c.finish_reason,
        valid_json=d.response is not None,
        legal=d.faction in result.legal,
        error=c.error or d.parse_error,
    )


def _narrator_record(turn: int, c: Completion) -> RequestRecord:
    return RequestRecord(
        turn=turn,
        actor="narrator",
        faction=None,
        prompt_tokens=c.prompt_tokens,
        output_tokens=c.completion_tokens,
        ttft_s=c.ttft_s,
        latency_s=c.latency_s,
        finish_reason=c.finish_reason,
        valid_json=None,
        legal=None,
        error=c.error,
    )


class Game:
    """One seeded game. All game randomness flows through self.rng."""

    def __init__(
        self,
        n_factions: int,
        seed: int,
        game_cfg: GameConfig,
        agent_cfg: AgentConfig,
        client: ChatClient,
        on_narration: Callable[[str], None] | None = None,
    ) -> None:
        self.seed = seed
        self.game_cfg = game_cfg
        self.agent_cfg = agent_cfg
        self.rng = random.Random(seed)
        self.world: World = generate_world(n_factions, game_cfg, self.rng)
        self.agents = Agents(client, game_cfg, agent_cfg)
        self.on_narration = on_narration
        self._pending: tuple[int, asyncio.Task[Completion]] | None = None

    @property
    def over(self) -> bool:
        return len(self.world.alive_factions()) <= 1

    async def _collect_narration(self) -> list[RequestRecord]:
        """Await an in-flight pipelined narration and append it to history."""
        if self._pending is None:
            return []
        turn, task = self._pending
        self._pending = None
        completion = await task
        if completion.error is None:
            self.world.history.append(HistoryEntry(turn=turn, summary=completion.text.strip()))
        return [_narrator_record(turn, completion)]

    async def play_turn(self) -> TurnRecord:
        world, cfg = self.world, self.game_cfg
        turn = world.turn
        start = time.perf_counter()

        shared = shared_prefix(world, cfg)
        alive = [f.id for f in world.alive_factions()]
        decisions = await asyncio.gather(
            *(
                self.agents.decide(
                    fid,
                    faction_messages(shared, world, fid, cfg),
                    request_seed(self.seed, turn, f"faction{fid}"),
                )
                for fid in alive
            )
        )
        decide_time = time.perf_counter() - start

        # Prompts for this turn are built, so the previous narration may now enter history.
        requests = await self._collect_narration()

        result = resolve_turn(world, {d.faction: d.response for d in decisions}, cfg, self.rng)
        requests.extend(_faction_record(turn, d, result) for d in decisions)
        statements = {d.faction: d.response.diplomatic_message for d in decisions if d.response}

        narration: str | None = None
        if self.agent_cfg.narrator != "off":
            # System message is the decision-time shared prefix, so it is already in the cache.
            messages = narrator_messages(shared, world, result, statements, cfg)
            coro = self.agents.narrate(
                messages, request_seed(self.seed, turn, "narrator"), self.on_narration
            )
            if self.agent_cfg.narrator == "sequential":
                completion = await coro
                requests.append(_narrator_record(turn, completion))
                if completion.error is None:
                    narration = completion.text.strip()
                    world.history.append(HistoryEntry(turn=turn, summary=narration))
            else:
                self._pending = (turn, asyncio.create_task(coro))

        return TurnRecord(
            turn=turn,
            n_factions=len(alive),
            wall_time_s=time.perf_counter() - start,
            decide_time_s=decide_time,
            requests=requests,
            result=result,
            outputs={d.faction: d.completion.text for d in decisions},
            narration=narration,
        )

    async def finish(self) -> list[RequestRecord]:
        """Drain any pipelined narration still in flight."""
        return await self._collect_narration()
