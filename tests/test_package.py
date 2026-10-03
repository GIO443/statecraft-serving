"""The repo is consumed as an installed package by speculative-statecraft."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

from bench.config import COMPOSE_FILE, REPO_ROOT


def test_installed_as_distribution() -> None:
    assert importlib.metadata.version("statecraft-serving") == "0.1.0"


def test_imports_work_outside_repo(tmp_path: Path) -> None:
    """No reliance on pytest's pythonpath: import from an unrelated working directory."""
    code = "import sim.engine, bench.harness, bench.config as c; print(c.REPO_ROOT)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert Path(out.stdout.strip()) == REPO_ROOT


def test_repo_root_resolves_configs() -> None:
    assert (REPO_ROOT / "configs" / "game" / "default.yaml").exists()
    assert COMPOSE_FILE.exists()
