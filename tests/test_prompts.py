"""Prompt layout tests. The prefix identity test guards prefix caching: if it fails, every
faction's prompt diverges early and the shared KV cache blocks stop being reused."""

from __future__ import annotations

import os
import random

import pytest

from sim.actions import ActionResponse
from sim.prompts import (
    Message,
    build_turn_prompts,
    narrator_messages,
    shared_prefix,
    static_section,
)
from sim.rules import resolve_turn
from sim.world import GameConfig, HistoryEntry, World, generate_world


def render_chatml(messages: list[Message]) -> str:
    """Approximation of the Qwen2.5 chat template (ChatML) with a generation prompt."""
    body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return body + "<|im_start|>assistant\n"


def common_prefix_len(texts: list[str]) -> int:
    return len(os.path.commonprefix(texts))


def _world(cfg: GameConfig, n_factions: int) -> World:
    world = generate_world(n_factions, cfg, random.Random(7))
    world.history = [HistoryEntry(turn=i, summary=f"summary {i}") for i in range(5)]
    return world


@pytest.mark.parametrize("n_factions", [4, 16, 64])
def test_shared_prefix_identical_across_factions(cfg: GameConfig, n_factions: int) -> None:
    world = _world(cfg, n_factions)
    prompts = build_turn_prompts(world, cfg)
    assert set(prompts) == set(range(n_factions))

    systems = {msgs[0]["content"] for msgs in prompts.values()}
    assert len(systems) == 1, "system message must be byte-identical for all factions"
    shared = systems.pop()
    assert "You are faction" not in shared

    rendered = [render_chatml(msgs) for msgs in prompts.values()]
    system_block = render_chatml([{"role": "system", "content": shared}])
    system_block = system_block.removesuffix("<|im_start|>assistant\n")
    assert common_prefix_len(rendered) >= len(system_block)
    # And the per-faction sections really differ.
    assert len({msgs[1]["content"] for msgs in prompts.values()}) == n_factions


def test_static_section_stable_across_turns(cfg: GameConfig) -> None:
    world = _world(cfg, 4)
    before = static_section(world, cfg)
    owned = world.owned(0)[0].id
    decisions = {
        f.id: ActionResponse.model_validate(
            {"action": {"type": "pass"}, "diplomatic_message": "hi"}
        )
        for f in world.factions
    }
    decisions[0] = ActionResponse.model_validate(
        {"action": {"type": "build", "province": owned, "kind": "fort"}, "diplomatic_message": ""}
    )
    resolve_turn(world, decisions, cfg, random.Random(0))
    world.history.append(HistoryEntry(turn=0, summary="a fort was built"))
    assert static_section(world, cfg) == before
    # The per-turn section did change, so the shared prefix is longer than the static part only.
    assert shared_prefix(world, cfg).startswith(before)


def test_prompts_are_deterministic(cfg: GameConfig) -> None:
    assert build_turn_prompts(_world(cfg, 8), cfg) == build_turn_prompts(_world(cfg, 8), cfg)


def test_history_window(cfg: GameConfig) -> None:
    world = _world(cfg, 4)
    text = shared_prefix(world, cfg)
    shown = [h.turn for h in world.history if f"Turn {h.turn}: {h.summary}" in text]
    assert shown == [h.turn for h in world.history[-cfg.history_turns :]]


def test_faction_section_contents(cfg: GameConfig) -> None:
    world = _world(cfg, 4)
    prompts = build_turn_prompts(world, cfg)
    for fid, msgs in prompts.items():
        user = msgs[1]["content"]
        faction = world.faction(fid)
        assert faction.name in user
        assert faction.personality in user
        for p in world.owned(fid):
            assert f"{p.id} {p.name} troops=" in user


def test_narrator_shares_prefix(cfg: GameConfig) -> None:
    world = _world(cfg, 4)
    shared = shared_prefix(world, cfg)
    result = resolve_turn(world, {}, cfg, random.Random(0))
    msgs = narrator_messages(shared, world, result, {0: "We come in peace.", 1: ""}, cfg)
    assert msgs[0]["content"] == shared
    assert "We come in peace." in msgs[1]["content"]
    assert f"at most {cfg.narrator_max_words} words" in msgs[1]["content"]
