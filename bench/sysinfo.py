"""Environment snapshot for env.json: GPU, driver, free VRAM, git state, image, caveats."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

from bench.config import REPO_ROOT

MIB_PER_GIB = 1024

WSL2_CAVEAT = (
    "Measured under WSL2 / Docker Desktop on Windows. Pinned host memory is unavailable under "
    "WSL (vLLM falls back to device memory for UVA buffers), and Windows reserves VRAM for the "
    "desktop; absolute numbers may differ from native Linux."
)


def _run(*cmd: str) -> str | None:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def gpu_info() -> dict[str, Any]:
    fields = "name,driver_version,memory.total,memory.used,memory.free"
    out = _run("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits")
    if not out:
        return {"error": "nvidia-smi unavailable"}
    name, driver, total, used, free = (s.strip() for s in out.splitlines()[0].split(","))
    return {
        "name": name,
        "driver_version": driver,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "memory_free_mib": int(free),
    }


def total_vram_gib(gpu: dict[str, Any]) -> float:
    return gpu["memory_total_mib"] / MIB_PER_GIB


def git_info() -> dict[str, Any]:
    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {"commit": commit or "no commits", "dirty": bool(status)}


def image_id(image: str) -> str | None:
    return _run("docker", "image", "inspect", "-f", "{{.Id}}", image)


def host_info() -> dict[str, Any]:
    return {"platform": platform.platform(), "python": platform.python_version()}
