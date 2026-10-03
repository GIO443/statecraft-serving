"""Prompt construction with a prefix-stable layout.

Layout (most to least stable):
  1. static_section: intro, rules, action format, static map. Identical for the whole game.
  2. turn_section: world state + recent history. Identical for every faction in a turn.
  3. faction_section: identity, personality, private state, instruction. Per faction.

Sections 1+2 form the system message, so after the chat template is applied every faction's
prompt shares a byte-identical prefix up to the end of the system turn. Nothing faction-specific,
no timestamps and no request ids may ever be put in sections 1 or 2.
"""

from __future__ import annotations

from collections.abc import Mapping

from sim.rules import TurnResult
from sim.world import GameConfig, World

Message = dict[str, str]

_INTRO = (
    "# Statecraft\n"
    "This is a turn-based grand strategy game. Factions control provinces on a map. Every turn "
    "all living factions choose one action simultaneously from the same world state. Actions "
    "then resolve in this order: treaty responses, treaty proposals, trade offers, builds, army "
    "moves, attacks (attacks in random order)."
)

_ACTION_FORMAT = """\
## Action format
Reply with exactly one JSON object and nothing else:
{"action": <action>, "diplomatic_message": <short public statement>}
<action> is one of:
{"type": "move_army", "from_province": <id>, "to_province": <id>, "troops": <n>}
{"type": "attack", "from_province": <id>, "to_province": <id>, "troops": <n>}
{"type": "build", "province": <id>, "kind": "troops" | "market" | "fort"}
{"type": "trade_offer", "to_faction": <id>, "gold": <n>}
{"type": "propose_treaty", "to_faction": <id>, "treaty": "peace" | "alliance" | "trade_pact"}
{"type": "respond_treaty", "proposal_id": <id>, "accept": true | false}
{"type": "pass"}
Example of a complete reply (both keys are always required):
{"action": {"type": "build", "province": 4, "kind": "fort"}, "diplomatic_message": "We fortify our borders and seek no quarrel."}"""  # noqa: E501


def _rules(cfg: GameConfig) -> str:
    variance_pct = round(cfg.combat_variance * 100)
    fort_pct = round(cfg.fort_defense_bonus * 100)
    return "\n".join(
        [
            "## Rules",
            f"- Income: each province yields its base income plus {cfg.market_income} gold per "
            f"market level. Each trade pact gives both parties {cfg.trade_pact_income} gold.",
            f"- Upkeep: 1 gold per {cfg.troops_per_upkeep_gold} troops per turn (rounded up). "
            "If you cannot pay, troops desert.",
            f"- Build troops: {cfg.troops_cost} gold for {cfg.troops_per_build} troops.",
            f"- Build market: {cfg.market_cost} gold, +{cfg.market_income} income, "
            f"max level {cfg.max_market_level}.",
            f"- Build fort: {cfg.fort_cost} gold, +{fort_pct}% defence per level, "
            f"max level {cfg.max_fort_level}.",
            "- move_army: move troops between two adjacent provinces you own.",
            "- attack: send troops from your province into an adjacent province you do not own. "
            "Attack strength is your troops; defence is the defending troops times the fort "
            f"bonus; each is scaled randomly by up to {variance_pct}%. If the attack is stronger "
            "you capture the province with the survivors, otherwise your attackers are lost.",
            "- Treaties: peace and alliance forbid attacks between the parties. Attacking a trade "
            "partner cancels the trade pact. A proposal must be accepted with respond_treaty "
            f"within {cfg.proposal_ttl_turns} turns.",
            "- trade_offer: give gold to another faction to improve relations.",
            "- A faction with no provinces is eliminated. Illegal actions count as pass.",
            f"- diplomatic_message: at most {cfg.max_message_chars} characters, seen by all.",
        ]
    )


def static_section(world: World, cfg: GameConfig) -> str:
    """Game-constant text. Depends only on config and static map properties."""
    map_lines = [
        "## Map",
        "Format: id name (base income) -> neighbour ids",
        *(
            f"{p.id} {p.name} ({p.income}) -> {','.join(map(str, world.adjacency[p.id]))}"
            for p in world.provinces
        ),
    ]
    return "\n\n".join([_INTRO, _rules(cfg), _ACTION_FORMAT, "\n".join(map_lines)])


def turn_section(world: World, cfg: GameConfig) -> str:
    """Per-turn state shared by all factions."""
    lines = [
        f"## Turn {world.turn}",
        "Factions (id name: treasury; provinces as id:troops with f=fort level, m=market level):",
    ]
    for faction in world.factions:
        if not faction.alive:
            lines.append(f"{faction.id} {faction.name}: eliminated")
            continue
        holdings = " ".join(
            f"{p.id}:{p.army}"
            + (f"f{p.fort}" if p.fort else "")
            + (f"m{p.market}" if p.market else "")
            for p in world.owned(faction.id)
        )
        lines.append(f"{faction.id} {faction.name}: {faction.treasury}g; {holdings}")
    lines.append("Treaties:")
    if world.treaties:
        lines.extend(f"{t.kind} {t.parties[0]}-{t.parties[1]}" for t in world.treaties)
    else:
        lines.append("none")
    lines.append("Recent history:")
    recent = world.history[-cfg.history_turns :] if cfg.history_turns else []
    if recent:
        lines.extend(f"Turn {h.turn}: {h.summary}" for h in recent)
    else:
        lines.append("none")
    return "\n".join(lines)


def shared_prefix(world: World, cfg: GameConfig) -> str:
    """System message content: identical for all factions (and the narrator) this turn."""
    return static_section(world, cfg) + "\n\n" + turn_section(world, cfg)


def faction_section(world: World, fid: int, cfg: GameConfig) -> str:
    faction = world.faction(fid)
    lines = [
        f"You are faction {fid}, {faction.name}: {faction.personality}.",
        f"Treasury: {faction.treasury} gold.",
        "Your provinces and their borders (neighbour id:owner faction id):",
    ]
    for p in world.owned(fid):
        borders = " ".join(f"{n}:{world.province(n).owner}" for n in world.adjacency[p.id])
        lines.append(f"{p.id} {p.name} troops={p.army} fort={p.fort} market={p.market} | {borders}")
    relations = " ".join(
        f"{oid}:{value}"
        for oid, value in sorted(faction.relations.items())
        if world.faction(oid).alive
    )
    lines.append(f"Relations (faction id:score): {relations}")
    pending = [p for p in world.proposals if p.to_faction == fid]
    if pending:
        lines.append("Treaty proposals to you:")
        lines.extend(f"proposal_id {p.id}: {p.kind} from faction {p.from_faction}" for p in pending)
    lines.append(
        "Choose your single action for this turn. Reply with only the JSON object, with both "
        '"action" and "diplomatic_message".'
    )
    return "\n".join(lines)


def faction_messages(shared: str, world: World, fid: int, cfg: GameConfig) -> list[Message]:
    return [
        {"role": "system", "content": shared},
        {"role": "user", "content": faction_section(world, fid, cfg)},
    ]


def build_turn_prompts(world: World, cfg: GameConfig) -> dict[int, list[Message]]:
    """Messages for every living faction; the system message object is shared verbatim."""
    shared = shared_prefix(world, cfg)
    return {f.id: faction_messages(shared, world, f.id, cfg) for f in world.alive_factions()}


def narrator_messages(
    shared: str,
    world: World,
    result: TurnResult,
    statements: Mapping[int, str],
    cfg: GameConfig,
) -> list[Message]:
    """Narrator reuses the decision-time shared prefix so its prompt hits the prefix cache too."""
    lines = [
        "You are the chronicler of this world. Summarize what happened this turn for the "
        f"players in at most {cfg.narrator_max_words} words of vivid prose.",
        "Resolved events:",
        *(result.events or ["nothing of note"]),
        "Public statements:",
        *(f"{world.faction(fid).name}: {text}" for fid, text in sorted(statements.items()) if text),
    ]
    return [{"role": "system", "content": shared}, {"role": "user", "content": "\n".join(lines)}]
