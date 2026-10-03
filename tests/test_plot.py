from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.plot_turns import load, main


def _write_variant(run: Path, name: str, started: str, base: float, stalled_turn: int = -1) -> Path:
    vdir = run / name
    vdir.mkdir(parents=True)
    (vdir / "env.json").write_text(json.dumps({"started_at": started}))
    rows = []
    for n in (4, 8):
        for repeat in (0, 1):
            for turn in range(3):
                stalled = n == 8 and repeat == 1 and turn == stalled_turn
                rows.append({
                    "n_factions": n, "repeat": repeat, "turn": turn, "warmup": turn == 0,
                    # turn 0 is warmup (100 s, excluded); a stalled turn takes 1200 s
                    "wall_time_s": 1200.0 if stalled else 100.0 if turn == 0 else base * n + repeat,
                    "request_errors": n if stalled else 0, "legal": n, "n_acting": n,
                })  # fmt: skip
    (vdir / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return vdir


def test_plot_turns(tmp_path: Path) -> None:
    run = tmp_path / "exp" / "20261002T000000Z"
    _write_variant(run, "slow", "2026-10-02T00:02:00", base=2.0)
    _write_variant(run, "fast", "2026-10-02T00:01:00", base=1.0)

    data = load(run)
    assert list(data) == ["fast", "slow"]  # ordered by start time, not name
    mean, sd, reps, excluded = data["fast"][4]
    assert (mean, reps, excluded) == (4.5, 2, 0)  # warmup excluded; repeats 4 and 5 -> 4.5
    assert sd > 0

    assert main([str(run)]) == 0
    assert (run / "seconds_per_turn.png").stat().st_size > 0
    with (run / "seconds_per_turn.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {r["variant"] for r in rows} == {"fast", "slow"}


def test_stalled_turns_excluded_and_partial_labeled(tmp_path: Path) -> None:
    run = tmp_path / "exp" / "20261002T000000Z"
    vdir = _write_variant(run, "prefix-off", "2026-10-02T00:01:00", base=1.0, stalled_turn=2)
    (vdir / "FAILED.md").write_text("stalled")
    data = load(run)
    assert list(data) == ["prefix-off (partial)"]
    mean, _, reps, excluded = data["prefix-off (partial)"][8]
    assert excluded == 1
    assert reps == 2
    assert mean == 8.5  # games average 8 and 9; the 1200 s stalled turn is not in it
