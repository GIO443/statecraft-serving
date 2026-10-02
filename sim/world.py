"""World state (provinces on a grid graph, factions, treaties) and seeded world generation."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

TreatyKind = Literal["peace", "alliance", "trade_pact"]
# Treaties that make attacks between the two parties illegal.
NON_AGGRESSION_TREATIES: frozenset[str] = frozenset({"peace", "alliance"})


class GameConfig(BaseModel):
    """Balance and generation parameters from configs/game/*.yaml. No defaults on purpose."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provinces_per_faction: int = Field(ge=1)
    diagonal_edge_fraction: float = Field(ge=0, le=1)
    province_income_min: int = Field(ge=0)
    province_income_max: int = Field(ge=0)
    starting_treasury: int = Field(ge=0)
    starting_army_per_province: int = Field(ge=0)
    troops_per_build: int = Field(ge=1)
    troops_cost: int = Field(ge=0)
    market_cost: int = Field(ge=0)
    market_income: int = Field(ge=0)
    max_market_level: int = Field(ge=0)
    fort_cost: int = Field(ge=0)
    fort_defense_bonus: float = Field(ge=0)
    max_fort_level: int = Field(ge=0)
    troops_per_upkeep_gold: int = Field(ge=1)
    combat_variance: float = Field(ge=0, lt=1)
    trade_pact_income: int = Field(ge=0)
    relation_start: int
    relation_min: int
    relation_max: int
    relation_delta_attack: int
    relation_delta_trade: int
    relation_delta_treaty: int
    proposal_ttl_turns: int = Field(ge=1)
    max_message_chars: int = Field(ge=1)
    history_turns: int = Field(ge=0)
    narrator_max_words: int = Field(ge=1)
    name_min_syllables: int = Field(ge=1)
    name_max_syllables: int = Field(ge=1)
    name_syllables: list[str] = Field(min_length=2)
    personalities: list[str] = Field(min_length=1)


def load_game_config(path: str | Path) -> GameConfig:
    with Path(path).open(encoding="utf-8") as f:
        return GameConfig.model_validate(yaml.safe_load(f))


class Province(BaseModel):
    id: int
    name: str
    owner: int
    income: int
    army: int
    fort: int = 0
    market: int = 0


class Faction(BaseModel):
    id: int
    name: str
    personality: str
    treasury: int
    relations: dict[int, int]
    alive: bool = True


class Treaty(BaseModel):
    kind: TreatyKind
    parties: tuple[int, int]  # sorted ascending


class Proposal(BaseModel):
    id: int
    kind: TreatyKind
    from_faction: int
    to_faction: int
    turn: int


class HistoryEntry(BaseModel):
    turn: int
    summary: str


class World(BaseModel):
    turn: int
    provinces: list[Province]  # province id == list index
    adjacency: dict[int, list[int]]
    factions: list[Faction]  # faction id == list index
    treaties: list[Treaty] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    next_proposal_id: int = 0
    history: list[HistoryEntry] = Field(default_factory=list)

    def province_exists(self, pid: int) -> bool:
        return 0 <= pid < len(self.provinces)

    def faction_exists(self, fid: int) -> bool:
        return 0 <= fid < len(self.factions)

    def province(self, pid: int) -> Province:
        return self.provinces[pid]

    def faction(self, fid: int) -> Faction:
        return self.factions[fid]

    def owned(self, fid: int) -> list[Province]:
        return [p for p in self.provinces if p.owner == fid]

    def alive_factions(self) -> list[Faction]:
        return [f for f in self.factions if f.alive]

    def are_adjacent(self, a: int, b: int) -> bool:
        return b in self.adjacency.get(a, ())

    def has_treaty(self, a: int, b: int, kinds: frozenset[str] | set[str]) -> bool:
        parties = (min(a, b), max(a, b))
        return any(t.parties == parties and t.kind in kinds for t in self.treaties)

    def treaties_of(self, fid: int) -> list[Treaty]:
        return [t for t in self.treaties if fid in t.parties]


def _grid_width(n: int) -> int:
    """Divisor of n closest to sqrt(n), so the grid is a full rectangle (ties: wider)."""
    root = math.sqrt(n)
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return min(divisors, key=lambda d: (abs(d - root), -d))


def _name_capacity(cfg: GameConfig) -> int:
    k = len(cfg.name_syllables)
    return sum(k**n for n in range(cfg.name_min_syllables, cfg.name_max_syllables + 1))


def _unique_name(rng: random.Random, cfg: GameConfig, taken: set[str]) -> str:
    while True:
        length = rng.randint(cfg.name_min_syllables, cfg.name_max_syllables)
        name = "".join(rng.choice(cfg.name_syllables) for _ in range(length)).capitalize()
        if name not in taken:
            taken.add(name)
            return name


def generate_world(n_factions: int, cfg: GameConfig, rng: random.Random) -> World:
    """Build a rectangular grid map; factions own contiguous strips along a snake path."""
    if n_factions < 2:
        raise ValueError("need at least 2 factions")
    n = n_factions * cfg.provinces_per_faction
    names_needed = n + n_factions
    if names_needed > _name_capacity(cfg) // 2:
        raise ValueError(f"name syllables too few for {names_needed} unique names")

    width = _grid_width(n)
    height = n // width
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}

    def link(a: int, b: int) -> None:
        adjacency[a].add(b)
        adjacency[b].add(a)

    for i in range(n):
        row, col = divmod(i, width)
        if col + 1 < width:
            link(i, i + 1)
        if row + 1 < height:
            link(i, i + width)
    diagonals = [
        (i, i + width + 1) for i in range(n) if i % width + 1 < width and i // width + 1 < height
    ]
    for a, b in rng.sample(diagonals, round(cfg.diagonal_edge_fraction * len(diagonals))):
        link(a, b)

    # Snake path through the grid: consecutive cells are always adjacent, so chunks are contiguous.
    snake: list[int] = []
    for row in range(height):
        cols = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
        snake.extend(row * width + col for col in cols)
    owner = {pid: idx // cfg.provinces_per_faction for idx, pid in enumerate(snake)}

    taken: set[str] = set()
    factions = [
        Faction(
            id=fid,
            name=_unique_name(rng, cfg, taken),
            personality=rng.choice(cfg.personalities),
            treasury=cfg.starting_treasury,
            relations={o: cfg.relation_start for o in range(n_factions) if o != fid},
        )
        for fid in range(n_factions)
    ]
    provinces = [
        Province(
            id=pid,
            name=_unique_name(rng, cfg, taken),
            owner=owner[pid],
            income=rng.randint(cfg.province_income_min, cfg.province_income_max),
            army=cfg.starting_army_per_province,
        )
        for pid in range(n)
    ]
    return World(
        turn=0,
        provinces=provinces,
        adjacency={pid: sorted(nbrs) for pid, nbrs in adjacency.items()},
        factions=factions,
    )
