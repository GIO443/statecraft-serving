"""Rule tests on the 2x2 small_world fixture (see conftest for the map and pinned balance)."""

from __future__ import annotations

import random

from sim.actions import ActionResponse
from sim.rules import TurnResult, resolve_turn
from sim.world import GameConfig, Proposal, Treaty, World

UPKEEP = 4  # each faction starts with 20 troops at 5 troops per gold


def act(**action: object) -> ActionResponse:
    return ActionResponse.model_validate({"action": action, "diplomatic_message": ""})


PASS = act(type="pass")


def run(
    world: World,
    cfg: GameConfig,
    f0: ActionResponse | None = PASS,
    f1: ActionResponse | None = PASS,
    seed: int = 0,
) -> TurnResult:
    return resolve_turn(world, {0: f0, 1: f1}, cfg, random.Random(seed))


def test_pass_turn_applies_upkeep_and_advances(small_world: World, rules_cfg: GameConfig) -> None:
    result = run(small_world, rules_cfg)
    assert small_world.turn == 1
    assert result.turn == 0
    assert result.legal == {0: "pass", 1: "pass"}
    assert [f.treasury for f in small_world.factions] == [50 - UPKEEP] * 2


def test_move_army(small_world: World, rules_cfg: GameConfig) -> None:
    run(small_world, rules_cfg, f0=act(type="move_army", from_province=0, to_province=1, troops=4))
    assert [small_world.province(0).army, small_world.province(1).army] == [6, 14]


def test_move_illegal_cases(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.province(3).owner = 0  # 0 and 3 are both f0's but not adjacent
    cases = {
        "not adjacent": act(type="move_army", from_province=0, to_province=3, troops=1),
        "not enough troops": act(type="move_army", from_province=0, to_province=1, troops=11),
        "province 2 is not yours": act(type="move_army", from_province=0, to_province=2, troops=1),
        "province 9 does not exist": act(
            type="move_army", from_province=9, to_province=1, troops=1
        ),
    }
    for reason, action in cases.items():
        world = small_world.model_copy(deep=True)
        result = run(world, rules_cfg, f0=action)
        assert len(result.illegal) == 1
        assert reason in result.illegal[0].reason
        assert 0 not in result.legal
        assert world.province(0).army == 10  # nothing executed


def test_attack_captures_undefended_province(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.province(2).army = 0
    small_world.treaties.append(Treaty(kind="trade_pact", parties=(0, 1)))
    run(small_world, rules_cfg, f0=act(type="attack", from_province=0, to_province=2, troops=6))
    assert small_world.province(2).owner == 0
    assert small_world.province(2).army == 6  # no defence, no losses
    assert small_world.province(0).army == 4
    assert small_world.faction(0).relations[1] == -30
    assert small_world.faction(1).relations[0] == -30
    assert small_world.treaties == []  # attacking a trade partner cancels the pact
    assert small_world.faction(1).alive


def test_attack_repelled(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.province(2).army = 100  # 110 troops -> 22 upkeep, affordable, so no desertion
    run(small_world, rules_cfg, f0=act(type="attack", from_province=0, to_province=2, troops=10))
    assert small_world.province(2).owner == 1
    assert small_world.province(0).army == 0
    # attack <= 12, defence >= 80, so the defender loses at most 15%
    assert 85 <= small_world.province(2).army < 100


def test_attack_own_province_illegal(small_world: World, rules_cfg: GameConfig) -> None:
    result = run(
        small_world, rules_cfg, f0=act(type="attack", from_province=0, to_province=1, troops=1)
    )
    assert result.illegal[0].reason == "cannot attack own province"


def test_peace_forbids_attack(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.treaties.append(Treaty(kind="peace", parties=(0, 1)))
    result = run(
        small_world, rules_cfg, f1=act(type="attack", from_province=2, to_province=0, troops=5)
    )
    assert "forbids attack" in result.illegal[0].reason
    assert small_world.province(0).owner == 0


def test_elimination(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.province(3).owner = 0
    small_world.province(2).army = 0
    small_world.proposals.append(Proposal(id=0, kind="peace", from_faction=1, to_faction=0, turn=0))
    result = run(
        small_world, rules_cfg, f0=act(type="attack", from_province=0, to_province=2, troops=5)
    )
    assert not small_world.faction(1).alive
    assert small_world.proposals == []
    assert any("eliminated" in e for e in result.events)
    # Eliminated factions no longer act or count as invalid.
    result = run(small_world, rules_cfg, f1=None)
    assert result.invalid_output == []
    assert set(result.legal) == {0}


def test_build(small_world: World, rules_cfg: GameConfig) -> None:
    run(small_world, rules_cfg, f0=act(type="build", province=0, kind="troops"))
    assert small_world.province(0).army == 20
    # 50 - 20 cost, then upkeep on 30 troops = 6
    assert small_world.faction(0).treasury == 50 - 20 - 6

    run(small_world, rules_cfg, f1=act(type="build", province=2, kind="market"))
    assert small_world.province(2).market == 1
    # turn 0: -4 upkeep; turn 1: -40 market, +3 market income, -4 upkeep
    assert small_world.faction(1).treasury == 50 - 4 - 40 + 3 - 4


def test_build_illegal(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.faction(0).treasury = 10
    small_world.province(2).fort = rules_cfg.max_fort_level
    result = run(
        small_world,
        rules_cfg,
        f0=act(type="build", province=0, kind="troops"),
        f1=act(type="build", province=2, kind="fort"),
    )
    assert {i.faction: i.reason for i in result.illegal} == {
        0: "not enough gold",
        1: "fort at max level",
    }


def test_trade_offer(small_world: World, rules_cfg: GameConfig) -> None:
    run(small_world, rules_cfg, f0=act(type="trade_offer", to_faction=1, gold=20))
    assert [f.treasury for f in small_world.factions] == [30 - UPKEEP, 70 - UPKEEP]
    assert small_world.faction(0).relations[1] == 5
    assert small_world.faction(1).relations[0] == 5


def test_trade_offer_illegal(small_world: World, rules_cfg: GameConfig) -> None:
    result = run(
        small_world,
        rules_cfg,
        f0=act(type="trade_offer", to_faction=0, gold=1),
        f1=act(type="trade_offer", to_faction=0, gold=51),
    )
    assert {i.faction: i.reason for i in result.illegal} == {
        0: "cannot target own faction",
        1: "not enough gold",
    }


def test_treaty_proposal_and_acceptance(small_world: World, rules_cfg: GameConfig) -> None:
    run(small_world, rules_cfg, f0=act(type="propose_treaty", to_faction=1, treaty="trade_pact"))
    assert len(small_world.proposals) == 1
    pid = small_world.proposals[0].id

    # Duplicate proposal and responding to a proposal addressed to someone else are illegal.
    result = run(
        small_world.model_copy(deep=True),
        rules_cfg,
        f0=act(type="propose_treaty", to_faction=1, treaty="trade_pact"),
        f1=act(type="respond_treaty", proposal_id=pid + 1, accept=True),
    )
    assert len(result.illegal) == 2
    result = run(
        small_world.model_copy(deep=True),
        rules_cfg,
        f0=act(type="respond_treaty", proposal_id=pid, accept=True),
    )
    assert "not addressed to you" in result.illegal[0].reason

    treasury = small_world.faction(1).treasury
    run(small_world, rules_cfg, f1=act(type="respond_treaty", proposal_id=pid, accept=True))
    assert small_world.treaties == [Treaty(kind="trade_pact", parties=(0, 1))]
    assert small_world.proposals == []
    assert small_world.faction(1).relations[0] == 15
    assert small_world.faction(1).treasury == treasury + 4 - UPKEEP  # pact income already paid


def test_treaty_rejection(small_world: World, rules_cfg: GameConfig) -> None:
    run(small_world, rules_cfg, f0=act(type="propose_treaty", to_faction=1, treaty="peace"))
    run(small_world, rules_cfg, f1=act(type="respond_treaty", proposal_id=0, accept=False))
    assert small_world.treaties == []
    assert small_world.proposals == []


def test_proposal_expires_after_ttl(small_world: World, rules_cfg: GameConfig) -> None:
    run(small_world, rules_cfg, f0=act(type="propose_treaty", to_faction=1, treaty="peace"))
    run(small_world, rules_cfg)  # turn 1: still answerable
    assert len(small_world.proposals) == 1
    run(small_world, rules_cfg)  # end of turn 2 = proposal turn 0 + ttl 2
    assert small_world.proposals == []


def test_action_fizzles_when_invalidated_mid_turn(
    small_world: World, rules_cfg: GameConfig
) -> None:
    """Responses resolve before attacks: f1 accepting f0's peace offer makes f0's same-turn
    attack fizzle."""
    small_world.proposals.append(Proposal(id=0, kind="peace", from_faction=0, to_faction=1, turn=0))
    small_world.next_proposal_id = 1
    small_world.province(2).army = 0
    result = run(
        small_world,
        rules_cfg,
        f0=act(type="attack", from_province=0, to_province=2, troops=5),
        f1=act(type="respond_treaty", proposal_id=0, accept=True),
    )
    # Legal against the snapshot, so not counted as illegal...
    assert result.illegal == []
    assert result.legal == {0: "attack", 1: "respond_treaty"}
    # ...but it fizzled at execution time.
    assert any("fizzled" in e for e in result.events)
    assert small_world.province(2).owner == 1


def test_invalid_output_counts_and_passes(small_world: World, rules_cfg: GameConfig) -> None:
    result = resolve_turn(small_world, {0: None}, rules_cfg, random.Random(0))  # f1 missing
    assert result.invalid_output == [0, 1]
    assert result.legal == {}


def test_desertion_when_upkeep_unpaid(small_world: World, rules_cfg: GameConfig) -> None:
    small_world.faction(0).treasury = 2  # upkeep 4 -> deficit 2 gold -> 10 troops desert
    result = run(small_world, rules_cfg)
    assert small_world.faction(0).treasury == 0
    assert small_world.province(0).army == 0  # largest army first, ties broken by lowest id
    assert small_world.province(1).army == 10
    assert any("deserted" in e for e in result.events)
