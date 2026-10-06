from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from bench.config import REPO_ROOT, load_experiment, pinned_image, resolve
from bench.harness import JsonlWriter, ServerStalled, guarded, run_variant, summarize
from bench.metrics import ProgressTracker, metric, parse_prometheus
from bench.predict import kv_bytes_per_token, predict_kv
from bench.server import docker_run_command, parse_startup_log, vllm_args
from bench.stats import percentile
from tests.fakes import FakeClient

EXPERIMENTS = sorted((REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
PHASE1 = REPO_ROOT / "configs" / "experiments" / "phase1-1.5b.yaml"
RTX_4070_LAPTOP_GIB = 8188 / 1024

# Verbatim lines from the vLLM v0.30.0 smoke-test container log (2026-10-02).
STARTUP_LOG = """\
(EngineCore pid=109) INFO 10-02 20:31:10 [model_runner.py:428] Model loading took 2.98 GiB memory and 65.554696 seconds
(EngineCore pid=109) INFO 10-02 20:31:34 [gpu_worker.py:640] Available KV cache memory: 1.95 GiB
(EngineCore pid=109) INFO 10-02 20:31:34 [kv_cache_utils.py:2395] GPU KV cache size: 72,992 tokens, Maximum concurrency for 4,096 tokens per request: 17.82x
"""  # noqa: E501


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.stem)
def test_experiment_configs_resolve(path: Path) -> None:
    exp = load_experiment(path)
    for variant in exp.variants:
        resolved = resolve(exp, variant)
        assert resolved.variant == variant.name
        assert resolved.run.server_warmup_seed not in resolved.run.seeds


def test_phase1_variants_change_one_thing() -> None:
    exp = load_experiment(PHASE1)
    runs = {v.name: resolve(exp, v) for v in exp.variants}
    default = runs["default"]
    assert default.server.extra_flags == ["--enable-prefix-caching"]
    assert runs["prefix-off"].server.extra_flags == ["--no-enable-prefix-caching"]
    assert runs["eager"].server.extra_flags == ["--enable-prefix-caching", "--enforce-eager"]
    assert runs["guided-off"].agent.guided_decoding is False
    assert runs["guided-off"].server == default.server
    for name, run in runs.items():
        changed = {
            k
            for k in ("server", "agent", "model", "run", "game")
            if getattr(run, k) != getattr(default, k)
        }
        assert len(changed) <= 1, (name, changed)


def test_warmup_seed_must_differ() -> None:
    exp = load_experiment(PHASE1)
    bad = exp.model_copy(update={"run": exp.run.model_copy(update={"server_warmup_seed": 0})})
    with pytest.raises(ValueError, match="server_warmup_seed"):
        resolve(bad, exp.variants[0])


def test_image_is_pinned(tmp_path: Path) -> None:
    assert "@sha256:" in pinned_image()
    compose = tmp_path / "compose.yml"
    compose.write_text("x-vllm-image: vllm/vllm-openai:latest\n")
    with pytest.raises(ValueError):
        pinned_image(compose)


def test_kv_prediction_reproduces_calibration() -> None:
    resolved = resolve(load_experiment(PHASE1), load_experiment(PHASE1).variants[0])
    assert kv_bytes_per_token(resolved.model) == 28 * 1024  # 28 KiB/token
    server = resolved.server.model_copy(update={"max_model_len": 4096})
    pred = predict_kv(resolved.model, server, RTX_4070_LAPTOP_GIB)
    assert pred.kv_cache_tokens == pytest.approx(72_992, rel=0.01)
    assert pred.max_concurrency == pytest.approx(17.82, rel=0.01)
    eager = server.model_copy(update={"extra_flags": ["--enforce-eager"]})
    eager_pred = predict_kv(resolved.model, eager, RTX_4070_LAPTOP_GIB)
    assert eager_pred.kv_cache_tokens == pytest.approx(112_704, rel=0.02)  # measured 2026-10-02


def test_parse_startup_log() -> None:
    info = parse_startup_log(STARTUP_LOG)
    assert info.weights_gib == 2.98
    assert info.kv_cache_gib == 1.95
    assert info.kv_cache_tokens == 72_992
    assert info.concurrency_tokens_per_request == 4096
    assert info.max_concurrency == 17.82
    assert parse_startup_log("nothing").kv_cache_tokens is None


def test_server_command() -> None:
    exp = load_experiment(PHASE1)
    resolved = resolve(exp, exp.variants[0])
    args = vllm_args(resolved.model, resolved.server)
    assert args[:2] == ["--model", "Qwen/Qwen2.5-1.5B-Instruct"]
    assert args[args.index("--max-model-len") + 1] == "16384"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.8"
    cmd = docker_run_command("img@sha256:x", resolved.model, resolved.server)
    assert cmd[:3] == ["docker", "run", "-d"]
    assert "hf-cache:/root/.cache/huggingface" in cmd
    assert cmd[cmd.index("img@sha256:x") + 1 :] == args
    assert cmd.count("-v") == 1  # no extra mounts unless configured


def test_server_extra_volumes() -> None:
    exp = load_experiment(PHASE1)
    resolved = resolve(exp, exp.variants[0])
    server = resolved.server.model_copy(update={"volumes": ["spec-data:/data"]})
    cmd = docker_run_command("img@sha256:x", resolved.model, server)
    i = cmd.index("spec-data:/data")
    assert cmd[i - 1] == "-v" and i < cmd.index("img@sha256:x")


def test_parse_prometheus() -> None:
    text = """\
# HELP vllm:prefix_cache_hits_total Prefix cache hits
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 120.0
vllm:prefix_cache_hits_created{engine="0",model_name="m"} 1.79e+09
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 200.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:e2e_request_latency_seconds_bucket{le="1.0",model_name="m"} 3.0
python_gc_objects_collected_total{generation="0"} 5.0
"""
    samples = parse_prometheus(text)
    assert len(samples) == 3
    assert metric(samples, "vllm:prefix_cache_hits_total") == 120.0
    assert metric(samples, "vllm:kv_cache_usage_perc") == 0.25
    assert metric(samples, "vllm:num_requests_running") is None


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([3.0], 99) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    assert percentile(list(range(101)), 99) == pytest.approx(99.0)


class FakeScraper:
    def __init__(self, stalled_s: float = 0.0) -> None:
        self.context: dict[str, Any] = {}
        self.scrapes: list[tuple[str, dict[str, Any]]] = []
        self.started = False
        self.stalled_s = stalled_s

    def start(self) -> None:
        self.started = True

    async def scrape(self, kind: str, **extra: Any) -> None:
        self.scrapes.append((kind, {**self.context, **extra}))

    def seconds_since_progress(self) -> float:
        return self.stalled_s


def _samples(gen: float, prompt: float, running: float, waiting: float) -> dict[str, float]:
    return {
        'vllm:generation_tokens_total{engine="0"}': gen,
        'vllm:prompt_tokens_total{engine="0"}': prompt,
        'vllm:num_requests_running{engine="0"}': running,
        'vllm:num_requests_waiting{engine="0"}': waiting,
    }


def test_progress_tracker_detects_stall() -> None:
    tracker = ProgressTracker()
    tracker.update(_samples(10, 100, 8, 56), now=0.0)
    tracker.update(_samples(20, 100, 8, 56), now=5.0)  # tokens moving: progress
    assert tracker.seconds_since_progress(now=5.0) == 0.0
    tracker.update(_samples(20, 100, 8, 56), now=200.0)  # frozen with work pending
    assert tracker.seconds_since_progress(now=200.0) == 195.0
    tracker.update(_samples(20, 100, 0, 0), now=300.0)  # idle is not a stall
    assert tracker.seconds_since_progress(now=300.0) == 0.0


def test_guarded_aborts_stalled_work() -> None:
    async def hang() -> None:
        await asyncio.sleep(3600)

    async def quick() -> int:
        return 7

    with pytest.raises(ServerStalled):
        asyncio.run(guarded(hang(), FakeScraper(stalled_s=999), stall_timeout_s=10, poll_s=0.01))
    assert asyncio.run(guarded(quick(), FakeScraper(), stall_timeout_s=10, poll_s=0.01)) == 7


def test_summarize_excludes_failed_turns(tmp_path: Path) -> None:
    rows = [
        {"n_factions": 4, "repeat": 0, "turn": 0, "warmup": True, "wall_time_s": 50.0,
         "request_errors": 0, "legal": 4, "n_acting": 4},
        {"n_factions": 4, "repeat": 0, "turn": 1, "warmup": False, "wall_time_s": 2.0,
         "request_errors": 0, "legal": 2, "n_acting": 4},
        {"n_factions": 4, "repeat": 0, "turn": 2, "warmup": False, "wall_time_s": 1200.0,
         "request_errors": 4, "legal": 0, "n_acting": 4},
    ]  # fmt: skip
    path = tmp_path / "turns.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = summarize(path)["4"]
    assert (s["seconds_per_turn_mean"], s["legal_rate"], s["excluded_turns"]) == (2.0, 0.5, 1)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_variant_writes_results(tmp_path: Path) -> None:
    exp = load_experiment(REPO_ROOT / "configs" / "experiments" / "smoke.yaml")
    resolved = resolve(exp, exp.variants[0])
    run = resolved.run
    client, scraper = FakeClient(), FakeScraper()
    writers = [
        JsonlWriter(tmp_path / f) for f in ("requests.jsonl", "turns.jsonl", "outputs.jsonl")
    ]
    asyncio.run(run_variant(resolved, client, scraper, *writers))
    for w in writers:
        w.close()

    turns = _read(tmp_path / "turns.jsonl")
    assert len(turns) == len(run.faction_counts) * run.repeats * run.turns
    assert {(t["n_factions"], t["repeat"], t["seed"]) for t in turns} == {
        (n, r, run.seeds[r]) for n in run.faction_counts for r in range(run.repeats)
    }
    assert all(t["warmup"] == (t["turn"] < run.warmup_turns) for t in turns)
    assert all(t["legal"] == t["n_acting"] == t["n_factions"] for t in turns)
    assert all(t["latency_p50_s"] is not None for t in turns)
    assert all(t["narrator_latency_s"] is not None for t in turns)

    requests = _read(tmp_path / "requests.jsonl")
    measured_factions = sum(t["n_acting"] for t in turns)
    assert sum(r["actor"] == "faction" for r in requests) == measured_factions
    assert sum(r["actor"] == "narrator" for r in requests) == len(turns)
    # Server warmup requests (seed 1000 map) are sent but not recorded.
    warmup_requests = run.server_warmup_factions * run.server_warmup_turns + run.server_warmup_turns
    assert len(client.requests) == len(requests) + warmup_requests

    outputs = _read(tmp_path / "outputs.jsonl")
    assert sum(o["actor"] == "faction" for o in outputs) == measured_factions
    assert sum(o["actor"] == "narrator" for o in outputs) == len(turns)
    assert all(o["text"] for o in outputs)

    assert scraper.started
    assert sum(kind == "turn_end" for kind, _ in scraper.scrapes) == len(turns)
    assert all(ctx["phase"] == "measure" for _, ctx in scraper.scrapes)

    summary = summarize(tmp_path / "turns.jsonl")
    assert set(summary) == {str(n) for n in run.faction_counts}
    assert all(s["repeats"] == run.repeats and s["legal_rate"] == 1.0 for s in summary.values())


def test_failed_server_start_removes_container(monkeypatch) -> None:
    from bench.server import VLLMServer

    exp = load_experiment(PHASE1)
    resolved = resolve(exp, exp.variants[0])
    server = VLLMServer("img@sha256:x", resolved.model, resolved.server)
    stopped: list[bool] = []

    def boom() -> None:
        raise RuntimeError("vLLM container exited during startup")

    monkeypatch.setattr(server, "start", boom)
    monkeypatch.setattr(server, "stop", lambda: stopped.append(True))
    with pytest.raises(RuntimeError, match="exited"), server:
        pass
    assert stopped == [True]


def test_startup_failure_surfaces_root_cause() -> None:
    from bench.server import startup_failure

    log = "\n".join(
        ["(EngineCore) ERROR Traceback (most recent call last):"]
        + ["(EngineCore) ERROR   frame"] * 500
        + ["(EngineCore) ERROR ValueError: To serve at least one request ... 0.3 GiB"]
        + ["(APIServer) noise"] * 500
        + ["(APIServer) RuntimeError: Engine core initialization failed."]
    )
    head = startup_failure(log, tail_chars=200).split("--- log tail ---")[0]
    assert "ValueError: To serve at least one request" in head
    assert "RuntimeError: Engine core initialization failed." in head


def test_world_factions_validated() -> None:
    exp = load_experiment(PHASE1)
    bad = exp.model_copy(update={"run": exp.run.model_copy(update={"world_factions": 32})})
    with pytest.raises(ValueError, match="world_factions"):
        resolve(bad, exp.variants[0])  # faction_counts go up to 64


def test_run_variant_fixed_world(tmp_path: Path) -> None:
    exp = load_experiment(REPO_ROOT / "configs" / "experiments" / "smoke.yaml")
    run = exp.run.model_copy(update={"world_factions": 8, "repeats": 1})
    resolved = resolve(exp.model_copy(update={"run": run}), exp.variants[0])
    client = FakeClient()
    writers = [
        JsonlWriter(tmp_path / f) for f in ("requests.jsonl", "turns.jsonl", "outputs.jsonl")
    ]
    asyncio.run(run_variant(resolved, client, FakeScraper(), *writers))
    for w in writers:
        w.close()
    turns = _read(tmp_path / "turns.jsonl")
    assert {t["world_factions"] for t in turns} == {8}
    assert all(t["n_acting"] == t["n_factions"] for t in turns)
    assert {t["n_factions"] for t in turns} == set(run.faction_counts)
    # Prompt size is set by the world, not by how many factions act.
    by_n = {}
    for r in _read(tmp_path / "requests.jsonl"):
        if r["actor"] == "faction" and r["turn"] == 0:
            by_n.setdefault(r["n_factions"], []).append(r["prompt_tokens"])
    assert len(by_n) == len(run.faction_counts)
    means = [sum(v) / len(v) for v in by_n.values()]
    assert max(means) - min(means) < 0.02 * max(means)
