from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.plot_turns import load, main


def _write_variant(run: Path, name: str, started: str, base: float) -> None:
    vdir = run / name
    vdir.mkdir(parents=True)
    (vdir / "env.json").write_text(json.dumps({"started_at": started}))
    rows = []
    for n in (4, 8):
        for repeat in (0, 1):
            for turn in range(3):
                wall = 100.0 if turn == 0 else base * n + repeat  # turn 0 is warmup
                rows.append({"n_factions": n, "repeat": repeat, "turn": turn,
                             "warmup": turn == 0, "wall_time_s": wall})  # fmt: skip
    (vdir / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_plot_turns(tmp_path: Path) -> None:
    run = tmp_path / "exp" / "20261002T000000Z"
    _write_variant(run, "slow", "2026-10-02T00:02:00", base=2.0)
    _write_variant(run, "fast", "2026-10-02T00:01:00", base=1.0)

    data = load(run)
    assert list(data) == ["fast", "slow"]  # ordered by start time, not name
    mean, sd, reps = data["fast"][4]
    assert (mean, reps) == (4.5, 2)  # warmup (100 s) excluded; repeats 4 and 5 -> 4.5
    assert sd > 0

    assert main([str(run)]) == 0
    assert (run / "seconds_per_turn.png").stat().st_size > 0
    with (run / "seconds_per_turn.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {r["variant"] for r in rows} == {"fast", "slow"}
