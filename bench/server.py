"""vLLM container lifecycle (docker run / health wait / logs / remove) and startup-log parsing."""

from __future__ import annotations

import re
import subprocess
import time

import httpx
from pydantic import BaseModel

from bench.config import ModelConfig, ServerConfig

CONTAINER_NAME = "statecraft-vllm"
HF_CACHE_VOLUME = "hf-cache"
HEALTH_POLL_S = 2.0


class StartupInfo(BaseModel):
    """What vLLM actually allocated, parsed from its startup log."""

    weights_gib: float | None
    kv_cache_gib: float | None
    kv_cache_tokens: int | None
    max_concurrency: float | None
    concurrency_tokens_per_request: int | None


_WEIGHTS = re.compile(r"Model loading took ([\d.]+) GiB")
_KV_MEM = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_KV_SIZE = re.compile(
    r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: "
    r"([\d.]+)x"
)


def _num(s: str) -> int:
    return int(s.replace(",", ""))


def parse_startup_log(text: str) -> StartupInfo:
    weights = _WEIGHTS.search(text)
    kv_mem = _KV_MEM.search(text)
    kv_size = _KV_SIZE.search(text)
    return StartupInfo(
        weights_gib=float(weights.group(1)) if weights else None,
        kv_cache_gib=float(kv_mem.group(1)) if kv_mem else None,
        kv_cache_tokens=_num(kv_size.group(1)) if kv_size else None,
        max_concurrency=float(kv_size.group(3)) if kv_size else None,
        concurrency_tokens_per_request=_num(kv_size.group(2)) if kv_size else None,
    )


def vllm_args(model: ModelConfig, server: ServerConfig) -> list[str]:
    return [
        "--model",
        model.model,
        *model.served_flags,
        "--gpu-memory-utilization",
        str(server.gpu_memory_utilization),
        "--max-model-len",
        str(server.max_model_len),
        *server.extra_flags,
    ]


def docker_run_command(image: str, model: ModelConfig, server: ServerConfig) -> list[str]:
    return [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{server.port}:8000",
        "-v", f"{HF_CACHE_VOLUME}:/root/.cache/huggingface",
        image,
        *vllm_args(model, server),
    ]  # fmt: skip


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, encoding="utf-8", check=check
    )


class VLLMServer:
    """Context manager owning the benchmark's vLLM container."""

    def __init__(self, image: str, model: ModelConfig, server: ServerConfig) -> None:
        self.image = image
        self.model = model
        self.server = server
        self.command = docker_run_command(image, model, server)

    def __enter__(self) -> VLLMServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _port_in_use(self) -> bool:
        try:
            httpx.get(f"{self.server.root_url}/health", timeout=2)
        except httpx.TransportError:
            return False
        return True

    def start(self) -> None:
        if self._port_in_use():
            raise RuntimeError(
                f"something already answers on port {self.server.port}; stop other vLLM "
                "containers first (serving and benchmarks must own the whole GPU)"
            )
        _docker("rm", "-f", CONTAINER_NAME, check=False)
        _docker(*self.command[1:])
        self.wait_ready()

    def running(self) -> bool:
        result = _docker("inspect", "-f", "{{.State.Running}}", CONTAINER_NAME, check=False)
        return result.stdout.strip() == "true"

    def wait_ready(self) -> None:
        deadline = time.monotonic() + self.server.startup_timeout_s
        while time.monotonic() < deadline:
            if not self.running():
                raise RuntimeError(f"vLLM container exited during startup:\n{self.logs()[-4000:]}")
            try:
                if httpx.get(f"{self.server.root_url}/health", timeout=5).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(HEALTH_POLL_S)
        raise TimeoutError(f"vLLM not healthy after {self.server.startup_timeout_s}s")

    def logs(self) -> str:
        result = _docker("logs", CONTAINER_NAME, check=False)
        return result.stdout + result.stderr

    def vllm_version(self) -> str:
        result = _docker(
            "exec", CONTAINER_NAME, "python3", "-c", "import vllm; print(vllm.__version__)",
            check=False,
        )  # fmt: skip
        return result.stdout.strip() or "unknown"

    def stop(self) -> None:
        _docker("rm", "-f", CONTAINER_NAME, check=False)
