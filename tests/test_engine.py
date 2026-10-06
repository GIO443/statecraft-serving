from __future__ import annotations

import asyncio
import json

import pytest

from bench.client import ChatRequest
from sim.agents import AgentConfig, NarratorMode
from sim.engine import Game, TurnRecord
from sim.world import GameConfig
from tests.fakes import NARRATION, FakeClient, faction_of, is_narrator, pass_response


def agent_cfg(**overrides: object) -> AgentConfig:
    base: dict[str, object] = {
        "faction_max_tokens": 128,
        "narrator_max_tokens": 160,
        "temperature": 0.7,
        "guided_decoding": True,
        "faction_priority": None,
        "narrator_priority": None,
        "narrator": "sequential",
    }
    return AgentConfig.model_validate(base | overrides)


def play(game: Game, turns: int) -> tuple[list[TurnRecord], list]:
    async def run() -> tuple[list[TurnRecord], list]:
        records = [await game.play_turn() for _ in range(turns)]
        return records, await game.finish()

    return asyncio.run(run())


def faction_requests(client: FakeClient, turn: int) -> list[ChatRequest]:
    return [
        r
        for r in client.requests
        if not is_narrator(r) and f"## Turn {turn}\n" in r.messages[0]["content"]
    ]


def test_sequential_turns(cfg: GameConfig) -> None:
    client = FakeClient()
    game = Game(4, seed=1, game_cfg=cfg, agent_cfg=agent_cfg(), client=client)
    records, tail = play(game, 3)
    assert tail == []
    for i, rec in enumerate(records):
        assert rec.turn == i
        assert rec.n_factions == 4
        assert [r.actor for r in rec.requests].count("faction") == 4
        assert [r.actor for r in rec.requests].count("narrator") == 1
        assert all(r.valid_json and r.legal for r in rec.requests if r.actor == "faction")
        assert rec.narration == NARRATION
        assert rec.wall_time_s >= rec.decide_time_s
    assert [h.turn for h in game.world.history] == [0, 1, 2]
    # Sequential: turn 1 prompts already contain turn 0's narration.
    assert all(
        f"Turn 0: {NARRATION}" in r.messages[0]["content"] for r in faction_requests(client, 1)
    )


def test_pipelined_narration_lags_one_turn(cfg: GameConfig) -> None:
    client = FakeClient()
    game = Game(4, seed=1, game_cfg=cfg, agent_cfg=agent_cfg(narrator="pipelined"), client=client)
    records, tail = play(game, 3)
    assert [r.turn for r in records[0].requests if r.actor == "narrator"] == []
    assert [r.turn for r in records[1].requests if r.actor == "narrator"] == [0]
    assert [r.turn for r in records[2].requests if r.actor == "narrator"] == [1]
    assert [r.turn for r in tail] == [2]
    assert all("Turn 0:" not in r.messages[0]["content"] for r in faction_requests(client, 1))
    assert all(
        f"Turn 0: {NARRATION}" in r.messages[0]["content"] for r in faction_requests(client, 2)
    )
    assert [h.turn for h in game.world.history] == [0, 1, 2]


def test_narrator_off(cfg: GameConfig) -> None:
    client = FakeClient()
    game = Game(4, seed=1, game_cfg=cfg, agent_cfg=agent_cfg(narrator="off"), client=client)
    records, _ = play(game, 2)
    assert not any(is_narrator(r) for r in client.requests)
    assert game.world.history == []
    assert all(rec.narration is None for rec in records)


@pytest.mark.parametrize("guided", [True, False])
def test_guided_decoding_flag(cfg: GameConfig, guided: bool) -> None:
    client = FakeClient()
    game = Game(4, seed=1, game_cfg=cfg, agent_cfg=agent_cfg(guided_decoding=guided), client=client)
    play(game, 1)
    for r in client.requests:
        if is_narrator(r):
            assert r.json_schema is None
        elif guided:
            assert r.json_schema is not None
            assert r.json_schema["properties"]["diplomatic_message"]["maxLength"] == (
                cfg.max_message_chars
            )
        else:
            assert r.json_schema is None


def test_priorities_and_shared_prefix(cfg: GameConfig) -> None:
    client = FakeClient()
    acfg = agent_cfg(faction_priority=0, narrator_priority=-1)
    play(Game(4, seed=1, game_cfg=cfg, agent_cfg=acfg, client=client), 1)
    narrator = [r for r in client.requests if is_narrator(r)]
    factions = [r for r in client.requests if not is_narrator(r)]
    assert [r.priority for r in narrator] == [-1]
    assert {r.priority for r in factions} == {0}
    # The narrator reuses the exact system message the factions used (prefix cache hit).
    assert {r.messages[0]["content"] for r in client.requests} == {
        narrator[0].messages[0]["content"]
    }


def test_invalid_and_failed_requests(cfg: GameConfig) -> None:
    def respond(request: ChatRequest) -> str:
        if not is_narrator(request) and faction_of(request) == 1:
            return '{"action": {"type": "pass"'  # truncated JSON
        return pass_response(request)

    client = FakeClient(
        respond=respond,
        error_for=lambda r: not is_narrator(r) and faction_of(r) == 2,
    )
    game = Game(4, seed=1, game_cfg=cfg, agent_cfg=agent_cfg(), client=client)
    (rec,), _ = play(game, 1)
    by_faction = {r.faction: r for r in rec.requests if r.actor == "faction"}
    assert by_faction[1].valid_json is False and by_faction[1].legal is False
    assert by_faction[1].error is not None and "invalid JSON" in by_faction[1].error
    assert by_faction[2].valid_json is False
    assert by_faction[2].error is not None and "APIConnectionError" in by_faction[2].error
    assert rec.result.invalid_output == [1, 2]
    assert by_faction[0].valid_json and by_faction[0].legal


def test_illegal_action_recorded(cfg: GameConfig) -> None:
    def respond(request: ChatRequest) -> str:
        if is_narrator(request):
            return NARRATION
        action = {"type": "trade_offer", "to_faction": faction_of(request), "gold": 1}
        return json.dumps({"action": action, "diplomatic_message": ""})

    (rec,), _ = play(Game(4, 1, cfg, agent_cfg(), FakeClient(respond=respond)), 1)
    assert all(r.valid_json and not r.legal for r in rec.requests if r.actor == "faction")
    assert len(rec.result.illegal) == 4


def test_faction_requests_run_concurrently(cfg: GameConfig) -> None:
    delay = 0.1
    client = FakeClient(delay_s=delay)
    game = Game(16, seed=1, game_cfg=cfg, agent_cfg=agent_cfg(narrator="off"), client=client)
    (rec,), _ = play(game, 1)
    assert rec.decide_time_s < 4 * delay  # serial would take 16 * delay


def test_narration_streams_to_callback(cfg: GameConfig) -> None:
    tokens: list[str] = []
    game = Game(4, 1, cfg, agent_cfg(), FakeClient(), on_narration=tokens.append)
    play(game, 1)
    assert "".join(tokens).strip() == NARRATION


def test_game_replays_identically(cfg: GameConfig) -> None:
    def respond(request: ChatRequest) -> str:
        if is_narrator(request):
            return NARRATION
        fid = faction_of(request)
        user = request.messages[-1]["content"]
        own = int(user.split("Your provinces and their borders")[1].split("\n")[1].split()[0])
        action = {"type": "build", "province": own, "kind": "troops"}
        return json.dumps({"action": action, "diplomatic_message": f"{fid} arms"})

    def run(mode: NarratorMode) -> str:
        game = Game(8, 5, cfg, agent_cfg(narrator=mode), FakeClient(respond=respond))
        play(game, 4)
        return game.world.model_dump_json()

    assert run("sequential") == run("sequential")
    assert run("pipelined") == run("pipelined")


def test_acting_factions_subset(cfg: GameConfig) -> None:
    """Control runs: full-size world (same prompts), but only some factions call the model."""
    full_client, sub_client = FakeClient(), FakeClient()
    full = Game(8, seed=3, game_cfg=cfg, agent_cfg=agent_cfg(), client=full_client)
    sub = Game(8, seed=3, game_cfg=cfg, agent_cfg=agent_cfg(), client=sub_client, acting_factions=3)
    (full_rec,), _ = play(full, 1)
    (rec,), _ = play(sub, 1)

    acting = sorted(faction_of(r) for r in faction_requests(sub_client, 0))
    assert acting == [0, 1, 2]
    assert rec.n_factions == 3
    assert [r.actor for r in rec.requests].count("faction") == 3
    # Passive factions pass legally; nothing is counted as invalid output.
    assert rec.result.invalid_output == [] and rec.result.illegal == []
    assert set(rec.result.legal) == set(range(8))
    # Same world, same prompts: the acting factions see byte-identical messages.
    full_msgs = {faction_of(r): r.messages for r in faction_requests(full_client, 0)}
    for r in faction_requests(sub_client, 0):
        assert r.messages == full_msgs[faction_of(r)]
    assert full_rec.n_factions == 8


@pytest.mark.parametrize("acting", [0, 9])
def test_acting_factions_out_of_range(cfg: GameConfig, acting: int) -> None:
    with pytest.raises(ValueError, match="acting_factions"):
        Game(
            8,
            seed=0,
            game_cfg=cfg,
            agent_cfg=agent_cfg(),
            client=FakeClient(),
            acting_factions=acting,
        )
