# CLAUDE.md

## Project overview

A turn-based grand strategy world simulation where many LLM-driven factions act every turn, served by vLLM on a single 8 GB consumer GPU. The simulation is the workload; the real goal is inference optimization.

Two pillars:

1. **Serving optimization.** Make a world turn as fast as possible as faction count scales (4 to 64), using prefix caching, guided JSON decoding, scheduling, quantization, and KV cache budgeting.
2. **Speculative decoding.** Train our own EAGLE-style draft head on the simulation's own outputs and show it beats generic speculation on this workload. Find the faction count where speculation stops helping (GPU becomes compute saturated) and explain why.

Every optimization must be measured, and every result must be explained in terms of mechanism (memory-bandwidth-bound decode, compute-bound prefill, KV cache capacity), not just reported.

## Hardware and environment

- Host: Windows, RTX 40-series laptop/desktop GPU, **8 GB VRAM** (Ada, so FP8 is native).
- GPU work runs in **Docker Desktop (WSL2 backend)**. vLLM runs in the `vllm/vllm-openai` image. Pin a specific version tag once chosen; never benchmark on `latest`.
- The simulation, agents, and benchmark harness run **natively on Windows** in a uv-managed Python 3.12 environment and talk to vLLM over HTTP at `http://localhost:8000/v1`.
- Shell is **PowerShell**. Write commands and scripts for PowerShell, not bash, unless they run inside a container.
- Model weights live in the named Docker volume `hf-cache`, mounted at `/root/.cache/huggingface`.
- Windows reserves some VRAM for the desktop. Default `--gpu-memory-utilization` to 0.8 and only raise it after checking free memory.

### Hard constraints

- **Serving and training never run at the same time.** Both need the whole GPU. Compose services use profiles so only one GPU service is up.
- Ask before downloading any model over ~2 GB.
- Benchmark numbers carry a "WSL2 / Docker on Windows" caveat (pinned memory is disabled under WSL). Record this in every results file.

## Current status

- Docker Desktop with GPU access is working.
- **Pinned vLLM: v0.30.0** (torch 2.13.0+cu130, transformers 5.17.0), pinned by digest in `docker-compose.yml`. Guided decoding is sent as `structured_outputs: {"json": schema}`; per-request `priority` needs `--scheduling-policy priority`.
- Phase 1 in progress: `sim/` (world, actions, rules, prompts, agents, engine) and `bench/client.py` are done and tested. Live tests: `$env:VLLM_BASE_URL = "http://localhost:8000/v1"; uv run pytest tests/test_live.py`.
- `bench/` harness done: `uv run python -m bench.harness configs/experiments/<exp>.yaml [--variant X] [--dry-run]` launches its own container (`statecraft-vllm`, fresh per variant), writes `results/<exp>/<UTC stamp>/<variant>/`. Plot: `uv run python -m analysis.plot_turns results/<exp>/<stamp>`.
- Prompt size grows with the world (measured with the Qwen2.5 tokenizer): ~1.35k tokens at 4 factions, ~2.5k at 16, ~4.1k at 32, ~7.9k at 64 (narrator ~12k at 64). Decision: keep the growth; sweeps use `max_model_len 16384` (v1 allocates KV on demand, so this reserves nothing).
- KV calibration (1.5B bf16, util 0.8, 8188 MiB RTX 4070 Laptop): weights 2.98 GiB, KV 1.95 GiB = 72,992 tokens at 28 KiB/token, CUDA graphs ~0.35 GiB.
- With Windows using ~1.4 GiB, free VRAM is ~6.55 GiB, exactly the 0.8 budget. Do not raise utilization without checking.
- Repo split (2026-10-02): this repo is **statecraft-serving** (Phase 1, github.com/GIO443/statecraft-serving, private). Phases 2-4 go in the sibling private repo **speculative-statecraft** (github.com/GIO443/speculative-statecraft, empty), which will depend on this one for `sim/` and `bench/`.
- First phase1-1.5b sweep (results/phase1-1.5b/20261002T*, commit 9743c87) is superseded. Findings kept: prefix caching ~6x at 64 factions (15.7 vs ~91 s/turn); 64-faction cost is superlinear because decode reads N x context KV per step (TPOT 17 -> 139 ms); vLLM stalled under KV pressure with prefix caching off (/health stayed 200); unguided 1.5B dropped `diplomatic_message` in 100% of replies.
- Fixes since: complete-reply example in the prompt (unguided validity 0% -> 86%), outputs.jsonl, first-error detail, stall watchdog (FAILED.md + vllm.log), failed turns excluded and counted, eager KV calibration (eager frees 1.08 GiB, not 0.35), `cascade-on` variant (`--no-disable-cascade-attn`, opt-in in v0.30).
- **Phase 1 sweep for 1.5B done** (results/phase1-1.5b/20261003T001956Z, commit da7df74, 0 failed/excluded turns). Seconds per turn at 4 / 16 / 64 factions: default 3.84 / 5.17 / 14.31; prefix-off 3.78 / 8.87 / 74.12 (5.2x at 64); guided-off 3.31 / 4.78 / 12.60 (grammar costs 7-19% TPOT; unguided valid 78-92%, legal-given-valid unchanged ~67%); eager 5.28 / 6.40 / 14.51 (launch overhead dominates small batches, converges at 64; extra KV unused since default peaks ~58%); cascade-on 3.55 / 4.55 / 12.62 (TPOT -8..-15% where >= 8 requests; 4-faction difference is run-to-run noise ~7%). Narrator hit its 200-token cap in ~46% of turns.
- **Next:** README "why this matters" (toy model of a multi-agent decision cycle on shared context) with these results; rename folder to statecraft-serving; make this repo installable for speculative-statecraft; model matrix (needs download approval).

## Repository layout (target)

```
.
├── CLAUDE.md
├── README.md                  # project story, results, plots (written last)
├── docker-compose.yml         # services: vllm (profile: serve), trainer (profile: train)
├── docker/
│   └── trainer.Dockerfile     # FROM pinned vllm image; adds training deps
├── pyproject.toml             # uv project for the native Windows side
├── configs/
│   ├── models/                # one YAML per model + quantization + vLLM flags
│   └── experiments/           # one YAML per experiment sweep
├── sim/
│   ├── world.py               # world state, provinces, factions, resources
│   ├── rules.py               # deterministic action resolution
│   ├── actions.py             # action schema (pydantic) + JSON schema export
│   ├── prompts.py             # prompt construction, prefix-stable layout
│   ├── agents.py              # async faction agents + narrator
│   └── engine.py              # turn loop
├── bench/
│   ├── client.py              # async OpenAI-compatible client with timing
│   ├── harness.py             # runs experiment configs, writes results
│   ├── metrics.py             # scrapes vLLM /metrics (Prometheus)
│   └── sysinfo.py             # nvidia-smi snapshot, versions, git hash
├── spec/
│   ├── collect.py             # log prompts + target completions to JSONL
│   ├── extract_hidden.py      # (trainer container) target hidden states to safetensors shards
│   ├── draft_head.py          # EAGLE-style draft head model
│   ├── train.py               # (trainer container) training loop
│   └── export.py              # export head in a format vLLM can load
├── analysis/                  # notebooks/scripts that turn results into plots
├── results/                   # JSONL results, one dir per run (gitignored except summaries)
└── tests/
```

## Simulation design

- **Deterministic and seeded.** Given a seed and the same LLM outputs, a game replays identically. All randomness goes through one seeded RNG.
- **World:** provinces as a graph (adjacency), each with owner, resources, army strength. Factions have treasury, armies, relations with every other faction, and a short personality string.
- **Actions** (validated by pydantic, exported as JSON schema for guided decoding): `move_army`, `attack`, `build`, `trade_offer`, `propose_treaty`, `respond_treaty`, `pass`. Each action response also includes a short `diplomatic_message` free-text field (capped length) so outputs mix structured and free text.
- **Turn loop:** all factions decide in parallel from the same world snapshot, then rules resolve actions deterministically, then the narrator writes a short turn summary.
- **Narrator** is player facing: it streams, and should run at higher priority than background factions.
- Invalid or illegal actions (valid JSON but breaking game rules) resolve to `pass` and are counted in results.

### Prompt layout (critical for prefix caching)

Order every faction prompt from most stable to least stable, and keep shared sections **byte identical** across factions:

1. Static: system prompt, game rules, action schema description, static map.
2. Per turn, shared: current world state and recent history summary, identical for all factions this turn.
3. Per faction: identity, personality, private state, and the instruction to act.

No timestamps, request IDs, or faction-specific content may appear before section 3. Add a test that asserts the shared prefix is identical across all factions for a given turn.

## Measurement

Every run writes a results directory containing:

- `config.yaml`: the exact merged config used.
- `env.json`: vLLM image tag, model, quantization, all vLLM flags, git hash, GPU name, driver version, free VRAM at start, and the WSL2 caveat.
- `requests.jsonl`: per request: faction, turn, prompt tokens, output tokens, TTFT, end-to-end latency, whether output was valid JSON and a legal action.
- `turns.jsonl`: per turn: wall time, p50/p99 per-agent latency, faction count.
- `server_metrics.jsonl`: periodic scrapes of vLLM `/metrics` (prefix cache hit rate, KV cache usage, running/waiting requests, spec decode acceptance when enabled).

Headline metric: **seconds per world turn** vs faction count. Secondary: p99 agent latency, tokens/sec, legal action rate, prefix cache hit rate, mean accepted draft length.

Run each config at least 3 times and report mean and spread. Warm up the server before timing.

## Phases

### Phase 1: Simulation and baseline serving

- Build `sim/` with unit tests for rules and the prefix identity test.
- Build `bench/` and run baseline sweeps: faction counts 4, 8, 16, 32, 64.
- Experiments: prefix caching on vs off; guided decoding on vs off (and its overhead); `max_model_len` vs max concurrency; CUDA graphs vs `--enforce-eager` (graphs cost VRAM that could go to KV cache).
- Model matrix: 1.5B FP16 baseline, 3B FP8, 3B AWQ 4-bit, 7B/8B AWQ 4-bit if it fits with usable KV cache.
- Before each model run, write down a predicted KV cache token budget and max concurrency, then compare with what vLLM logs at startup. Keep these predictions in the results.

**Done when:** one command runs a full sweep and produces results for every config, and a script plots seconds per turn vs faction count.

### Phase 2: Training data collection

- Pick the target model for speculation (likely 3B FP8; confirm it leaves room for a draft head).
- Run many games across seeds and faction counts; log every prompt and the target's completion.
- In the trainer container, run the target model with Hugging Face transformers to extract hidden states for those sequences; write sharded safetensors to a shared volume. Stream to disk; do not hold the dataset in RAM.

### Phase 3: Draft head

- Implement a minimal EAGLE-style draft head ourselves (one transformer layer predicting next tokens from target hidden states plus token embeddings) so the mechanism is understood, not copied.
- Train in the trainer container. Hold out games by seed for validation; track acceptance-style metrics offline.
- Export in a format the pinned vLLM version can load as an EAGLE drafter. **Check the installed vLLM version's docs and source for the expected format and for compatibility with guided decoding and prefix caching before building the exporter.** Document any incompatibilities as findings.

### Phase 4: Analysis

Compare: no speculation, n-gram (prompt lookup) speculation, a public generic drafter for the target model if one exists, and our draft head.

- Mean accepted length broken down by output region: JSON scaffolding, `diplomatic_message` text, narrator text.
- Speedup vs faction count; locate the crossover where speculation stops helping and explain it (spare compute at low batch vs saturation at high batch).
- Stretch: enable speculation dynamically based on load.

## Working conventions

- Small, reviewable steps. Propose a plan before large changes.
- Python 3.12, type hints throughout, pydantic for schemas, `ruff` for lint and format, `pytest` for tests.
- Async clients (`openai` AsyncOpenAI or `httpx`) for concurrent agent requests.
- All experiment parameters come from YAML configs; no magic numbers in code.
- Never silently change a benchmark setting between compared runs. If something must change, record it in `env.json`.
- When unsure whether a vLLM flag or feature exists in the pinned version, check the version's docs or source instead of guessing.