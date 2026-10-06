"""Run an experiment: for each variant launch vLLM, warm up, sweep faction counts x repeats.

    uv run python -m bench.harness configs/experiments/phase1-1.5b.yaml [--variant NAME] [--dry-run]

Each variant gets a fresh server (so no prefix cache carries over between compared runs) and
its own results directory with config.yaml, env.json, requests.jsonl, turns.jsonl,
outputs.jsonl, server_metrics.jsonl and summary.json. A variant whose server stalls is
aborted, logged in FAILED.md with the container log, and the sweep moves on.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
from collections import defaultdict
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TextIO

import yaml

from bench.client import ChatClient, OpenAIChatClient
from bench.config import REPO_ROOT, ResolvedRun, load_experiment, pinned_image, resolve
from bench.metrics import MetricsScraper
from bench.predict import predict_kv
from bench.server import VLLMServer, parse_startup_log, vllm_args
from bench.stats import percentile
from bench.sysinfo import WSL2_CAVEAT, git_info, gpu_info, host_info, image_id, total_vram_gib
from sim.engine import Game, RequestRecord, TurnRecord


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._f: TextIO = path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps(row, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class Scraper(Protocol):
    context: dict[str, Any]

    def start(self) -> None: ...
    async def scrape(self, kind: str, **extra: Any) -> Any: ...
    def seconds_since_progress(self) -> float: ...


class ServerStalled(RuntimeError):
    """vLLM answers /health but has made no token progress with requests pending."""


async def guarded[T](
    work: Awaitable[T], scraper: Scraper, stall_timeout_s: float, poll_s: float
) -> T:
    """Await work, but abort it if the server stops making progress."""
    task = asyncio.ensure_future(work)
    while True:
        done, _ = await asyncio.wait({task}, timeout=poll_s)
        if done:
            return task.result()
        stalled = scraper.seconds_since_progress()
        if stalled > stall_timeout_s:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise ServerStalled(f"no token progress for {stalled:.0f}s with requests pending")


def output_rows(meta: dict[str, Any], rec: TurnRecord) -> list[dict[str, Any]]:
    """Raw completions: for diagnosing format failures and as Phase 2 training data."""
    rows = [
        {**meta, "turn": rec.turn, "actor": "faction", "faction": fid, "text": text}
        for fid, text in sorted(rec.outputs.items())
    ]
    if rec.narration is not None:
        rows.append(
            {**meta, "turn": rec.turn, "actor": "narrator", "faction": None, "text": rec.narration}
        )
    return rows


def request_row(meta: dict[str, Any], r: RequestRecord, warmup_turns: int) -> dict[str, Any]:
    return {**meta, "warmup": r.turn < warmup_turns, **r.model_dump()}


def turn_row(meta: dict[str, Any], rec: TurnRecord, warmup_turns: int) -> dict[str, Any]:
    factions = [r for r in rec.requests if r.actor == "faction"]
    latencies = [r.latency_s for r in factions]
    ttfts = [r.ttft_s for r in factions if r.ttft_s is not None]
    output_tokens = sum(r.output_tokens or 0 for r in factions)
    narrator = [r for r in rec.requests if r.actor == "narrator" and r.turn == rec.turn]
    return {
        **meta,
        "turn": rec.turn,
        "warmup": rec.turn < warmup_turns,
        "n_acting": rec.n_factions,
        "wall_time_s": rec.wall_time_s,
        "decide_time_s": rec.decide_time_s,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p99_s": percentile(latencies, 99),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p99_s": percentile(ttfts, 99),
        "prompt_tokens": sum(r.prompt_tokens or 0 for r in factions),
        "output_tokens": output_tokens,
        "decode_tokens_per_s": output_tokens / rec.decide_time_s if rec.decide_time_s else None,
        "valid_json": sum(bool(r.valid_json) for r in factions),
        "legal": sum(bool(r.legal) for r in factions),
        "illegal": len(rec.result.illegal),
        "invalid_output": len(rec.result.invalid_output),
        "request_errors": sum(r.finish_reason is None for r in factions),
        "narrator_latency_s": narrator[0].latency_s if narrator else None,
    }


async def run_variant(
    resolved: ResolvedRun,
    client: ChatClient,
    scraper: Scraper,
    requests_out: JsonlWriter,
    turns_out: JsonlWriter,
    outputs_out: JsonlWriter,
) -> None:
    run = resolved.run

    def guard[T](work: Awaitable[T]) -> Awaitable[T]:
        return guarded(work, scraper, run.stall_timeout_s, run.metrics_interval_s)

    scraper.context = {"phase": "server_warmup"}
    scraper.start()
    warm = Game(
        run.server_warmup_factions, run.server_warmup_seed, resolved.game, resolved.agent, client
    )
    for _ in range(run.server_warmup_turns):
        await guard(warm.play_turn())
    await guard(warm.finish())

    for n in run.faction_counts:
        for repeat in range(run.repeats):
            meta = {
                "experiment": resolved.experiment,
                "variant": resolved.variant,
                "n_factions": n,
                "world_factions": run.world_factions or n,
                "repeat": repeat,
                "seed": run.seeds[repeat],
            }
            scraper.context = {"phase": "measure", **meta}
            game = Game(
                run.world_factions or n,
                run.seeds[repeat],
                resolved.game,
                resolved.agent,
                client,
                acting_factions=n if run.world_factions else None,
            )
            for _ in range(run.turns):
                if game.over:
                    break
                rec = await guard(game.play_turn())
                for r in rec.requests:
                    requests_out.write(request_row(meta, r, run.warmup_turns))
                turns_out.write(turn_row(meta, rec, run.warmup_turns))
                for row in output_rows(meta, rec):
                    outputs_out.write(row)
                await scraper.scrape("turn_end", turn=rec.turn)
            for r in await guard(game.finish()):
                requests_out.write(request_row(meta, r, run.warmup_turns))
            print(f"  {resolved.variant}: {n} factions, repeat {repeat} done", flush=True)


def usable(row: dict[str, Any]) -> bool:
    """Turns that count toward timing summaries: not warmup, no failed requests."""
    return not row["warmup"] and row["request_errors"] == 0


def summarize(turns_path: Path) -> dict[str, Any]:
    """Mean of per-game mean seconds/turn with spread across repeats.

    Warmup turns are excluded; turns with failed requests (timeouts, server stalls) are
    excluded and counted in excluded_turns, never silently dropped.
    """
    per_game: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    legal: dict[int, list[float]] = defaultdict(list)
    excluded: dict[int, int] = defaultdict(int)
    with turns_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n = row["n_factions"]
            if not usable(row):
                excluded[n] += not row["warmup"]
                continue
            per_game[n][row["repeat"]].append(row["wall_time_s"])
            legal[n].append(row["legal"] / row["n_acting"])
    summary: dict[str, Any] = {}
    for n in sorted(set(per_game) | set(excluded)):
        means = [statistics.fmean(v) for v in per_game[n].values()]
        summary[str(n)] = {
            "seconds_per_turn_mean": statistics.fmean(means) if means else None,
            "seconds_per_turn_stdev": statistics.stdev(means) if len(means) > 1 else 0.0,
            "repeats": len(means),
            "legal_rate": statistics.fmean(legal[n]) if legal[n] else None,
            "excluded_turns": excluded[n],
        }
    return summary


async def _measure(resolved: ResolvedRun, run_dir: Path) -> None:
    writers = [JsonlWriter(run_dir / f) for f in ("requests.jsonl", "turns.jsonl", "outputs.jsonl")]
    metrics_out = JsonlWriter(run_dir / "server_metrics.jsonl")
    client = OpenAIChatClient(
        resolved.server.base_url, resolved.model.model, resolved.run.request_timeout_s
    )
    scraper = MetricsScraper(
        f"{resolved.server.root_url}/metrics", resolved.run.metrics_interval_s, metrics_out.write
    )
    try:
        await run_variant(resolved, client, scraper, *writers)
    finally:
        await scraper.stop()
        await client.aclose()
        for w in [*writers, metrics_out]:
            w.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--variant", action="append", help="run only these variants")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true", help="print plan and predictions only")
    args = parser.parse_args(argv)

    exp = load_experiment(args.experiment)
    variants = [v for v in exp.variants if not args.variant or v.name in args.variant]
    if not variants:
        parser.error(f"no variants match {args.variant}")
    image = pinned_image()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = args.results_dir / exp.name / stamp

    for variant in variants:
        resolved = resolve(exp, variant)
        gpu = gpu_info()
        prediction = predict_kv(resolved.model, resolved.server, total_vram_gib(gpu))
        server = VLLMServer(image, resolved.model, resolved.server)
        p = prediction
        print(
            f"[{variant.name}] predicted KV cache: {p.kv_cache_tokens:,} tokens "
            f"({p.kv_cache_gib:.2f} GiB), max concurrency {p.max_concurrency:.1f}x "
            f"at {resolved.server.max_model_len} tokens"
        )
        if args.dry_run:
            print("  " + " ".join(server.command))
            continue

        run_dir = root / variant.name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        started = datetime.now(UTC).isoformat()
        with server:
            startup = parse_startup_log(server.logs())
            env = {
                "started_at": started,
                "caveat": WSL2_CAVEAT,
                "vllm_image": image,
                "vllm_image_id": image_id(image),
                "vllm_version": server.vllm_version(),
                "model": resolved.model.model,
                "quantization_flags": resolved.model.served_flags,
                "vllm_flags": vllm_args(resolved.model, resolved.server),
                "docker_command": server.command,
                "git": git_info(),
                "gpu_at_start": gpu,
                "host": host_info(),
                "kv_prediction": prediction.model_dump(),
                "kv_actual": startup.model_dump(),
            }
            (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
            if startup.kv_cache_tokens is not None:
                err = startup.kv_cache_tokens / max(prediction.kv_cache_tokens, 1) - 1
                print(
                    f"  actual KV cache: {startup.kv_cache_tokens:,} tokens "
                    f"({startup.kv_cache_gib} GiB), prediction error {err:+.1%}"
                )
            try:
                asyncio.run(_measure(resolved, run_dir))
            except ServerStalled as e:
                note = f"Variant aborted at {datetime.now(UTC).isoformat()}: {e}. See vllm.log.\n"
                (run_dir / "FAILED.md").write_text(note, encoding="utf-8")
                print(f"  FAILED: {e}", flush=True)
            finally:
                # Saved before the container is removed: the only record of server-side issues.
                (run_dir / "vllm.log").write_text(server.logs(), encoding="utf-8")

        summary = summarize(run_dir / "turns.jsonl")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        for n, s in summary.items():
            mean, sd = s["seconds_per_turn_mean"], s["seconds_per_turn_stdev"]
            timing = f"{mean:.2f} ± {sd:.2f} s/turn" if mean is not None else "no usable turns"
            legal = f"{s['legal_rate']:.0%}" if s["legal_rate"] is not None else "n/a"
            print(f"  {n:>3} factions: {timing}, legal {legal}, excluded {s['excluded_turns']}")
    if not args.dry_run:
        print(f"results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
