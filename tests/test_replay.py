"""Determinism: same seed + same agent outputs => identical game, turn by turn."""

from __future__ import annotations

import random

from sim.actions import ACTION_TYPES, ActionResponse
from sim.prompts import build_turn_prompts
from sim.rules import resolve_turn
from sim.world import GameConfig, HistoryEntry, World, generate_world

TURNS = 15


def random_response(world: World, fid: int, rng: random.Random) -> ActionResponse | None:
    """Scripted stand-in for the LLM: mostly plausible, sometimes illegal, sometimes unparsable."""
    kind = rng.choice([*ACTION_TYPES, "invalid"])
    n_p, n_f = len(world.provinces), len(world.factions)
    own = [p.id for p in world.owned(fid)] or [0]
    src = rng.choice(own)
    dst = rng.choice(world.adjacency[src])
    actions: dict[str, dict[str, object]] = {
        "move_army": {"from_province": src, "to_province": dst, "troops": rng.randint(1, 8)},
        "attack": {"from_province": src, "to_province": dst, "troops": rng.randint(1, 15)},
        "build": {"province": rng.randrange(n_p), "kind": rng.choice(["troops", "market", "fort"])},
        "trade_offer": {"to_faction": rng.randrange(n_f), "gold": rng.randint(1, 30)},
        "propose_treaty": {
            "to_faction": rng.randrange(n_f),
            "treaty": rng.choice(["peace", "alliance", "trade_pact"]),
        },
        "respond_treaty": {
            "proposal_id": rng.randrange(world.next_proposal_id + 1),
            "accept": rng.random() < 0.5,
        },
        "pass": {},
    }
    if kind == "invalid":
        return None
    return ActionResponse.model_validate(
        {"action": {"type": kind, **actions[kind]}, "diplomatic_message": f"f{fid}"}
    )


def play(cfg: GameConfig, seed: int, n_factions: int) -> tuple[str, list[str], list[str]]:
    game_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    world = generate_world(n_factions, cfg, game_rng)
    results, prompts = [], []
    for _ in range(TURNS):
        prompts.append(repr(build_turn_prompts(world, cfg)))
        decisions = {f.id: random_response(world, f.id, policy_rng) for f in world.alive_factions()}
        result = resolve_turn(world, decisions, cfg, game_rng)
        world.history.append(HistoryEntry(turn=result.turn, summary=" ".join(result.events)[:200]))
        results.append(result.model_dump_json())
    return world.model_dump_json(), results, prompts


def test_replay_is_identical(cfg: GameConfig) -> None:
    assert play(cfg, seed=3, n_factions=8) == play(cfg, seed=3, n_factions=8)


def test_different_seed_diverges(cfg: GameConfig) -> None:
    assert play(cfg, seed=3, n_factions=8)[0] != play(cfg, seed=4, n_factions=8)[0]


def test_random_play_exercises_rules(cfg: GameConfig) -> None:
    """Sanity: a random game hits legal, illegal and invalid paths and keeps invariants."""
    game_rng, policy_rng = random.Random(11), random.Random(12)
    world = generate_world(8, cfg, game_rng)
    legal = illegal = invalid = 0
    for _ in range(TURNS):
        decisions = {f.id: random_response(world, f.id, policy_rng) for f in world.alive_factions()}
        result = resolve_turn(world, decisions, cfg, game_rng)
        legal += len(result.legal)
        illegal += len(result.illegal)
        invalid += len(result.invalid_output)
        assert all(p.army >= 0 for p in world.provinces)
        assert all(f.treasury >= 0 for f in world.factions)
        assert all(world.faction(p.owner).alive for p in world.provinces)
    assert legal and illegal and invalid
