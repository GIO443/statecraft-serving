"""Live tests against a running vLLM server. Opt-in: set VLLM_BASE_URL (and VLLM_MODEL).

$env:VLLM_BASE_URL = "http://localhost:8000/v1"; uv run pytest tests/test_live.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from bench.client import OpenAIChatClient
from sim.agents import AgentConfig
from sim.engine import Game, TurnRecord
from sim.world import GameConfig

BASE_URL = os.environ.get("VLLM_BASE_URL")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TIMEOUT_S = 120.0

pytestmark = pytest.mark.skipif(BASE_URL is None, reason="VLLM_BASE_URL not set")


def _play_one_turn(cfg: GameConfig, guided: bool) -> TurnRecord:
    acfg = AgentConfig(
        faction_max_tokens=160,
        narrator_max_tokens=200,
        temperature=0.7,
        guided_decoding=guided,
        faction_priority=None,
        narrator_priority=None,
        narrator="sequential",
    )

    async def run() -> TurnRecord:
        assert BASE_URL is not None
        client = OpenAIChatClient(BASE_URL, MODEL, TIMEOUT_S)
        try:
            return await Game(4, 0, cfg, acfg, client).play_turn()
        finally:
            await client.aclose()

    return asyncio.run(run())


def test_guided_turn_produces_valid_actions(cfg: GameConfig) -> None:
    rec = _play_one_turn(cfg, guided=True)
    factions = [r for r in rec.requests if r.actor == "faction"]
    assert len(factions) == 4
    for r in factions:
        assert r.error is None, r.error
        assert r.valid_json, rec.outputs[r.faction]
        assert r.prompt_tokens and r.output_tokens and r.ttft_s is not None
    narrator = [r for r in rec.requests if r.actor == "narrator"]
    assert narrator[0].error is None and rec.narration


def test_unguided_turn_runs(cfg: GameConfig) -> None:
    """Unguided output may be invalid JSON; that is a measured outcome, not a failure."""
    rec = _play_one_turn(cfg, guided=False)
    assert all(r.finish_reason is not None for r in rec.requests)  # every request completed
