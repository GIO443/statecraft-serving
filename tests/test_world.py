from __future__ import annotations

import random
from collections import deque

import pytest

from sim.world import GameConfig, World, generate_world

FACTION_COUNTS = [2, 4, 8, 16, 32, 64]


def _connected(world: World, nodes: set[int]) -> bool:
    start = next(iter(nodes))
    seen = {start}
    queue = deque([start])
    while queue:
        for n in world.adjacency[queue.popleft()]:
            if n in nodes and n not in seen:
                seen.add(n)
                queue.append(n)
    return seen == nodes


def test_generation_is_deterministic(cfg: GameConfig) -> None:
    a = generate_world(8, cfg, random.Random(42))
    b = generate_world(8, cfg, random.Random(42))
    c = generate_world(8, cfg, random.Random(43))
    assert a.model_dump_json() == b.model_dump_json()
    assert a.model_dump_json() != c.model_dump_json()


@pytest.mark.parametrize("n_factions", FACTION_COUNTS)
def test_world_structure(cfg: GameConfig, n_factions: int) -> None:
    world = generate_world(n_factions, cfg, random.Random(n_factions))
    n_provinces = n_factions * cfg.provinces_per_faction
    assert len(world.provinces) == n_provinces
    assert len(world.factions) == n_factions
    assert [p.id for p in world.provinces] == list(range(n_provinces))
    assert [f.id for f in world.factions] == list(range(n_factions))

    # Adjacency is symmetric with no self loops, and the whole map is connected.
    for pid, nbrs in world.adjacency.items():
        assert pid not in nbrs
        for n in nbrs:
            assert pid in world.adjacency[n]
    assert _connected(world, set(range(n_provinces)))

    # Every faction owns an equal, contiguous territory.
    for faction in world.factions:
        owned = {p.id for p in world.owned(faction.id)}
        assert len(owned) == cfg.provinces_per_faction
        assert _connected(world, owned)
        assert set(faction.relations) == set(range(n_factions)) - {faction.id}

    names = [p.name for p in world.provinces] + [f.name for f in world.factions]
    assert len(names) == len(set(names))
    for p in world.provinces:
        assert cfg.province_income_min <= p.income <= cfg.province_income_max


def test_rejects_single_faction(cfg: GameConfig) -> None:
    with pytest.raises(ValueError):
        generate_world(1, cfg, random.Random(0))


def test_world_json_round_trip(cfg: GameConfig) -> None:
    world = generate_world(4, cfg, random.Random(1))
    assert World.model_validate_json(world.model_dump_json()) == world
