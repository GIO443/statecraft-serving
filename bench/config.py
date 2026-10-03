"""Experiment configuration: model YAML + experiment YAML (with variants) -> resolved runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sim.agents import AgentConfig
from sim.world import GameConfig, load_game_config

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Architecture(_Strict):
    num_layers: int
    num_kv_heads: int
    head_dim: int
    kv_dtype_bytes: int


class ModelConfig(_Strict):
    """One model + quantization. Memory figures are the written-down prediction inputs."""

    name: str
    model: str  # Hugging Face id passed to --model
    served_flags: list[str]  # dtype / quantization flags
    architecture: Architecture
    weights_gib: float  # expected "Model loading took X GiB"
    overhead_gib: float  # memory used in every mode (activations floor, non-torch)
    eager_savings_gib: float  # released by --enforce-eager (CUDA graphs + compile-mode memory)
    notes: str


class ServerConfig(_Strict):
    port: int
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int
    extra_flags: list[str]
    startup_timeout_s: float

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def root_url(self) -> str:
        return f"http://localhost:{self.port}"


class RunConfig(_Strict):
    faction_counts: list[int]
    repeats: int = Field(ge=1)
    seeds: list[int]  # seeds[repeat]; identical across variants so comparisons are paired
    turns: int = Field(ge=1)  # turns per game
    warmup_turns: int = Field(ge=0)  # first turns of each game, flagged warmup in results
    server_warmup_factions: int  # one untimed game after server start (graphs, caches, JIT)
    server_warmup_turns: int
    server_warmup_seed: int  # must not be in seeds, or warmup pre-fills the measured prefix cache
    request_timeout_s: float
    metrics_interval_s: float
    stall_timeout_s: float  # abort the variant if no token progress this long with work pending


class Variant(_Strict):
    name: str
    server: dict[str, Any] = Field(default_factory=dict)  # overrides ServerConfig fields
    extra_flags: list[str] = Field(default_factory=list)  # appended to server.extra_flags
    agent: dict[str, Any] = Field(default_factory=dict)  # overrides AgentConfig fields


class ExperimentConfig(_Strict):
    name: str
    description: str
    model_config_path: str
    game_config_path: str
    server: ServerConfig
    agent: AgentConfig
    run: RunConfig
    variants: list[Variant] = Field(min_length=1)


class ResolvedRun(_Strict):
    """Everything one variant needs; dumped verbatim to config.yaml."""

    experiment: str
    variant: str
    model: ModelConfig
    server: ServerConfig
    agent: AgentConfig
    run: RunConfig
    game: GameConfig


def _load_yaml(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml(path))


def resolve(exp: ExperimentConfig, variant: Variant) -> ResolvedRun:
    if len(exp.run.seeds) < exp.run.repeats:
        raise ValueError("need at least one seed per repeat")
    if exp.run.server_warmup_seed in exp.run.seeds[: exp.run.repeats]:
        raise ValueError("server_warmup_seed must differ from measured seeds")
    server = exp.server.model_dump() | variant.server
    server["extra_flags"] = [*server["extra_flags"], *variant.extra_flags]
    game_path = Path(exp.game_config_path)
    return ResolvedRun(
        experiment=exp.name,
        variant=variant.name,
        model=load_model_config(exp.model_config_path),
        server=ServerConfig.model_validate(server),
        agent=AgentConfig.model_validate(exp.agent.model_dump() | variant.agent),
        run=exp.run,
        game=load_game_config(game_path if game_path.is_absolute() else REPO_ROOT / game_path),
    )


def pinned_image(compose_file: Path = COMPOSE_FILE) -> str:
    """The vLLM image is pinned in exactly one place: docker-compose.yml's x-vllm-image."""
    image = _load_yaml(compose_file)["x-vllm-image"]
    if "@sha256:" not in image and image.endswith(":latest"):
        raise ValueError("refusing to benchmark on an unpinned :latest image")
    return image
