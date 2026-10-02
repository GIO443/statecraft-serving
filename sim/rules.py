"""Deterministic action resolution.

Legality is judged against the start-of-turn snapshot (what the faction saw); illegal actions
resolve to pass and are counted. Legal actions then execute in a fixed phase order and are
re-checked at execution time: an action invalidated by an earlier action this turn "fizzles"
(reported as an event, not counted as illegal, since the model could not have known).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping

from pydantic import BaseModel, Field

from sim.actions import (
    Action,
    ActionResponse,
    Attack,
    Build,
    MoveArmy,
    Pass,
    ProposeTreaty,
    RespondTreaty,
    TradeOffer,
)
from sim.world import NON_AGGRESSION_TREATIES, GameConfig, Proposal, Treaty, World

# Execution order of action types within a turn (attacks are additionally shuffled).
PHASE_ORDER: tuple[str, ...] = (
    "respond_treaty",
    "propose_treaty",
    "trade_offer",
    "build",
    "move_army",
    "attack",
)


class IllegalAction(BaseModel):
    faction: int
    action: str
    reason: str


class TurnResult(BaseModel):
    turn: int
    events: list[str] = Field(default_factory=list)
    illegal: list[IllegalAction] = Field(default_factory=list)
    invalid_output: list[int] = Field(default_factory=list)  # factions whose output didn't parse
    legal: dict[int, str] = Field(default_factory=dict)  # faction -> legal action type


def build_cost(kind: str, cfg: GameConfig) -> int:
    return {"troops": cfg.troops_cost, "market": cfg.market_cost, "fort": cfg.fort_cost}[kind]


def _check_partner(world: World, fid: int, other: int) -> str | None:
    if not world.faction_exists(other):
        return f"faction {other} does not exist"
    if other == fid:
        return "cannot target own faction"
    if not world.faction(other).alive:
        return f"faction {other} is eliminated"
    return None


def _check_own(world: World, fid: int, pid: int) -> str | None:
    if not world.province_exists(pid):
        return f"province {pid} does not exist"
    if world.province(pid).owner != fid:
        return f"province {pid} is not yours"
    return None


def check_legal(world: World, fid: int, action: Action, cfg: GameConfig) -> str | None:
    """Return None if the action is legal for faction fid in this world state, else a reason."""
    match action:
        case MoveArmy():
            reason = _check_own(world, fid, action.from_province) or _check_own(
                world, fid, action.to_province
            )
            if reason:
                return reason
            if not world.are_adjacent(action.from_province, action.to_province):
                return "provinces are not adjacent"
            if action.troops > world.province(action.from_province).army:
                return "not enough troops"
        case Attack():
            if reason := _check_own(world, fid, action.from_province):
                return reason
            if not world.province_exists(action.to_province):
                return f"province {action.to_province} does not exist"
            if not world.are_adjacent(action.from_province, action.to_province):
                return "provinces are not adjacent"
            defender = world.province(action.to_province).owner
            if defender == fid:
                return "cannot attack own province"
            if action.troops > world.province(action.from_province).army:
                return "not enough troops"
            if world.has_treaty(fid, defender, NON_AGGRESSION_TREATIES):
                return f"treaty with faction {defender} forbids attack"
        case Build():
            if reason := _check_own(world, fid, action.province):
                return reason
            if build_cost(action.kind, cfg) > world.faction(fid).treasury:
                return "not enough gold"
            province = world.province(action.province)
            if action.kind == "market" and province.market >= cfg.max_market_level:
                return "market at max level"
            if action.kind == "fort" and province.fort >= cfg.max_fort_level:
                return "fort at max level"
        case TradeOffer():
            if reason := _check_partner(world, fid, action.to_faction):
                return reason
            if action.gold > world.faction(fid).treasury:
                return "not enough gold"
        case ProposeTreaty():
            if reason := _check_partner(world, fid, action.to_faction):
                return reason
            if world.has_treaty(fid, action.to_faction, {action.treaty}):
                return "treaty already exists"
            if any(
                p.from_faction == fid
                and p.to_faction == action.to_faction
                and p.kind == action.treaty
                for p in world.proposals
            ):
                return "identical proposal already pending"
        case RespondTreaty():
            proposal = _find_proposal(world, action.proposal_id)
            if proposal is None:
                return f"proposal {action.proposal_id} does not exist"
            if proposal.to_faction != fid:
                return f"proposal {action.proposal_id} is not addressed to you"
        case Pass():
            pass
    return None


def _find_proposal(world: World, proposal_id: int) -> Proposal | None:
    return next((p for p in world.proposals if p.id == proposal_id), None)


def _adjust_relation(world: World, a: int, b: int, delta: int, cfg: GameConfig) -> None:
    for x, y in ((a, b), (b, a)):
        relations = world.faction(x).relations
        relations[y] = max(cfg.relation_min, min(cfg.relation_max, relations[y] + delta))


def _name(world: World, fid: int) -> str:
    return world.faction(fid).name


def _exec_respond(
    world: World, fid: int, a: RespondTreaty, cfg: GameConfig, rng: random.Random, events: list[str]
) -> None:
    proposal = _find_proposal(world, a.proposal_id)
    assert proposal is not None
    world.proposals.remove(proposal)
    other = proposal.from_faction
    if not a.accept:
        events.append(f"{_name(world, fid)} rejected {_name(world, other)}'s {proposal.kind}.")
        return
    parties = (min(fid, other), max(fid, other))
    if not world.has_treaty(fid, other, {proposal.kind}):
        world.treaties.append(Treaty(kind=proposal.kind, parties=parties))
    _adjust_relation(world, fid, other, cfg.relation_delta_treaty, cfg)
    events.append(f"{_name(world, fid)} accepted {_name(world, other)}'s {proposal.kind}.")


def _exec_propose(
    world: World, fid: int, a: ProposeTreaty, cfg: GameConfig, rng: random.Random, events: list[str]
) -> None:
    world.proposals.append(
        Proposal(
            id=world.next_proposal_id,
            kind=a.treaty,
            from_faction=fid,
            to_faction=a.to_faction,
            turn=world.turn,
        )
    )
    world.next_proposal_id += 1
    events.append(f"{_name(world, fid)} proposed a {a.treaty} to {_name(world, a.to_faction)}.")


def _exec_trade(
    world: World, fid: int, a: TradeOffer, cfg: GameConfig, rng: random.Random, events: list[str]
) -> None:
    world.faction(fid).treasury -= a.gold
    world.faction(a.to_faction).treasury += a.gold
    _adjust_relation(world, fid, a.to_faction, cfg.relation_delta_trade, cfg)
    events.append(f"{_name(world, fid)} sent {a.gold} gold to {_name(world, a.to_faction)}.")


def _exec_build(
    world: World, fid: int, a: Build, cfg: GameConfig, rng: random.Random, events: list[str]
) -> None:
    province = world.province(a.province)
    world.faction(fid).treasury -= build_cost(a.kind, cfg)
    if a.kind == "troops":
        province.army += cfg.troops_per_build
    elif a.kind == "market":
        province.market += 1
    else:
        province.fort += 1
    events.append(f"{_name(world, fid)} built {a.kind} in {province.name}.")


def _exec_move(
    world: World, fid: int, a: MoveArmy, cfg: GameConfig, rng: random.Random, events: list[str]
) -> None:
    src, dst = world.province(a.from_province), world.province(a.to_province)
    src.army -= a.troops
    dst.army += a.troops
    events.append(f"{_name(world, fid)} moved {a.troops} troops from {src.name} to {dst.name}.")


def _exec_attack(
    world: World, fid: int, a: Attack, cfg: GameConfig, rng: random.Random, events: list[str]
) -> None:
    src, dst = world.province(a.from_province), world.province(a.to_province)
    defender = dst.owner
    src.army -= a.troops
    v = cfg.combat_variance
    attack = a.troops * rng.uniform(1 - v, 1 + v)
    defense = dst.army * (1 + dst.fort * cfg.fort_defense_bonus) * rng.uniform(1 - v, 1 + v)
    if attack > defense:
        dst.owner = fid
        dst.army = max(1, round(a.troops * (1 - defense / attack)))
        events.append(
            f"{_name(world, fid)} captured {dst.name} from {_name(world, defender)} "
            f"({dst.army} troops survive)."
        )
    else:
        dst.army = max(0, round(dst.army * (1 - attack / defense)))
        events.append(
            f"{_name(world, defender)} repelled {_name(world, fid)}'s attack on {dst.name} "
            f"({a.troops} attackers lost)."
        )
    _adjust_relation(world, fid, defender, cfg.relation_delta_attack, cfg)
    pact = (min(fid, defender), max(fid, defender))
    world.treaties = [
        t for t in world.treaties if not (t.kind == "trade_pact" and t.parties == pact)
    ]


_EXECUTORS: dict[str, Callable[..., None]] = {
    "respond_treaty": _exec_respond,
    "propose_treaty": _exec_propose,
    "trade_offer": _exec_trade,
    "build": _exec_build,
    "move_army": _exec_move,
    "attack": _exec_attack,
}


def _eliminate(world: World, events: list[str]) -> None:
    owners = {p.owner for p in world.provinces}
    for faction in world.alive_factions():
        if faction.id not in owners:
            faction.alive = False
            fid = faction.id
            world.treaties = [t for t in world.treaties if fid not in t.parties]
            world.proposals = [
                p for p in world.proposals if fid not in (p.from_faction, p.to_faction)
            ]
            events.append(f"{faction.name} has been eliminated.")


def _economy(world: World, cfg: GameConfig, events: list[str]) -> None:
    for faction in world.alive_factions():
        owned = world.owned(faction.id)
        pacts = sum(1 for t in world.treaties_of(faction.id) if t.kind == "trade_pact")
        income = sum(p.income + p.market * cfg.market_income for p in owned)
        income += pacts * cfg.trade_pact_income
        upkeep = math.ceil(sum(p.army for p in owned) / cfg.troops_per_upkeep_gold)
        faction.treasury += income - upkeep
        if faction.treasury < 0:
            deserters = -faction.treasury * cfg.troops_per_upkeep_gold
            faction.treasury = 0
            lost = 0
            for province in sorted(owned, key=lambda p: (-p.army, p.id)):
                take = min(province.army, deserters - lost)
                province.army -= take
                lost += take
                if lost >= deserters:
                    break
            events.append(f"{faction.name} could not pay upkeep; {lost} troops deserted.")


def _expire_proposals(world: World, cfg: GameConfig) -> None:
    world.proposals = [p for p in world.proposals if world.turn < p.turn + cfg.proposal_ttl_turns]


def resolve_turn(
    world: World,
    decisions: Mapping[int, ActionResponse | None],
    cfg: GameConfig,
    rng: random.Random,
) -> TurnResult:
    """Apply one turn of decisions to world in place. Missing/None decisions count as invalid."""
    result = TurnResult(turn=world.turn)
    accepted: dict[int, Action] = {}
    for faction in world.alive_factions():
        response = decisions.get(faction.id)
        if response is None:
            result.invalid_output.append(faction.id)
            continue
        reason = check_legal(world, faction.id, response.action, cfg)
        if reason is not None:
            result.illegal.append(
                IllegalAction(faction=faction.id, action=response.action.type, reason=reason)
            )
            continue
        accepted[faction.id] = response.action
        result.legal[faction.id] = response.action.type

    for phase in PHASE_ORDER:
        batch = [(fid, a) for fid, a in sorted(accepted.items()) if a.type == phase]
        if phase == "attack":
            rng.shuffle(batch)
        for fid, action in batch:
            reason = check_legal(world, fid, action, cfg)
            if reason is not None:
                result.events.append(f"{_name(world, fid)}'s {action.type} fizzled: {reason}.")
                continue
            _EXECUTORS[phase](world, fid, action, cfg, rng, result.events)

    _eliminate(world, result.events)
    _economy(world, cfg, result.events)
    _expire_proposals(world, cfg)
    world.turn += 1
    return result
