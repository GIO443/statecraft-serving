from __future__ import annotations

import random
from pathlib import Path

import pytest

from sim.world import GameConfig, World, generate_world, load_game_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "game" / "default.yaml"


@pytest.fixture
def cfg() -> GameConfig:
    return load_game_config(DEFAULT_CONFIG)


@pytest.fixture
def rules_cfg(cfg: GameConfig) -> GameConfig:
    """Default config with the balance values that rule tests assert on pinned explicitly."""
    return cfg.model_copy(
        update={
            "provinces_per_faction": 2,
            "diagonal_edge_fraction": 0.0,
            "starting_treasury": 50,
            "starting_army_per_province": 10,
            "troops_per_build": 10,
            "troops_cost": 20,
            "market_cost": 40,
            "market_income": 3,
            "max_market_level": 3,
            "fort_cost": 30,
            "troops_per_upkeep_gold": 5,
            "trade_pact_income": 4,
            "relation_start": 0,
            "relation_delta_attack": -30,
            "relation_delta_trade": 5,
            "relation_delta_treaty": 15,
            "proposal_ttl_turns": 2,
        }
    )


@pytest.fixture
def small_world(rules_cfg: GameConfig) -> World:
    """2 factions on a 2x2 grid with zero province income:

        0(f0) - 1(f0)
          |       |
        2(f1) - 3(f1)

    Each province has 10 troops, so each faction pays ceil(20/5) = 4 gold upkeep per turn.
    """
    world = generate_world(2, rules_cfg, random.Random(0))
    for province in world.provinces:
        province.income = 0
    assert [p.owner for p in world.provinces] == [0, 0, 1, 1]
    assert world.adjacency == {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
    return world
