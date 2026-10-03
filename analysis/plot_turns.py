"""Plot seconds per world turn vs faction count, one line per variant.

    uv run --group analysis python -m analysis.plot_turns results/phase1-1.5b/<timestamp>

Reads <run>/<variant>/turns.jsonl and summarizes it exactly like the harness: warmup turns and
turns with failed requests are excluded (the CSV counts the latter), each game is averaged, then
mean and standard deviation across repeats. Writes seconds_per_turn.png and .csv into <run>.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

from bench.harness import summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-8 in fixed order (validated: light mode, surface #fcfcfb). Slots 3 and 4
# are below 3:1 contrast, so every line is also direct-labeled and a CSV table is written.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cc2", "#e34b4b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)  # fmt: skip


Point = tuple[float, float, int, int]  # mean s/turn, stdev across repeats, repeats, excluded
Data = dict[str, dict[int, Point]]
# Marker files written by the harness (FAILED.md) or by hand (INCOMPLETE.md).
PARTIAL_MARKERS = ("FAILED.md", "INCOMPLETE.md")


def load(run_dir: Path) -> Data:
    """{variant label: {n_factions: Point}} in run order, using the harness's summarize()."""
    out: Data = {}

    def started(p: Path) -> str:
        env = p / "env.json"
        return json.loads(env.read_text(encoding="utf-8"))["started_at"] if env.exists() else ""

    # Run order == config order, so each variant keeps its color across plots.
    variant_dirs = sorted(
        (p for p in run_dir.iterdir() if (p / "turns.jsonl").exists()), key=lambda p: started(p)
    )
    for vdir in variant_dirs:
        partial = any((vdir / m).exists() for m in PARTIAL_MARKERS)
        label = f"{vdir.name} (partial)" if partial else vdir.name
        out[label] = {
            int(n): (
                s["seconds_per_turn_mean"],
                s["seconds_per_turn_stdev"],
                s["repeats"],
                s["excluded_turns"],
            )
            for n, s in summarize(vdir / "turns.jsonl").items()
            if s["seconds_per_turn_mean"] is not None
        }
    return out


def write_csv(data: Data, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_factions", "seconds_per_turn_mean", "seconds_per_turn_stdev",
                    "repeats", "excluded_turns"])  # fmt: skip
        for variant, points in data.items():
            for n, (mean, sd, reps, excluded) in points.items():
                w.writerow([variant, n, f"{mean:.4f}", f"{sd:.4f}", reps, excluded])


def plot(data: Data, title: str, path: Path) -> None:
    if len(data) > len(SERIES):
        raise ValueError("more variants than categorical slots; split into small multiples")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (variant, points) in enumerate(data.items()):
        xs = list(points)
        means = [points[n][0] for n in xs]
        sds = [points[n][1] for n in xs]
        ax.errorbar(
            xs, means, yerr=sds, color=SERIES[i], marker=MARKERS[i], markersize=6, linewidth=2,
            capsize=3, elinewidth=1, label=variant,
            markeredgecolor=SURFACE, markeredgewidth=1.5,  # surface ring on overlapping marks
        )  # fmt: skip
        ax.annotate(
            variant, (xs[-1], means[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", color=INK_2, fontsize=9,
        )  # fmt: skip
    all_n = sorted({n for points in data.values() for n in points})
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_n, [str(n) for n in all_n])
    ax.set_xlim(all_n[0] / 1.2, all_n[-1] * 1.6)  # room for direct labels
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Factions", color=INK_2)
    ax.set_ylabel("Seconds per world turn", color=INK_2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
    if len(data) >= 2:
        legend = ax.legend(frameon=False, loc="upper left", fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_2)
    fig.text(0.01, 0.01, "Mean ± sd across repeats; warmup turns excluded. WSL2 / Docker on "
             "Windows.", color=MUTED, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results/<experiment>/<timestamp>")
    args = parser.parse_args(argv)
    data = load(args.run_dir)
    if not data:
        parser.error(f"no turns.jsonl under {args.run_dir}")
    write_csv(data, args.run_dir / "seconds_per_turn.csv")
    title = f"Seconds per world turn — {args.run_dir.parent.name}"
    plot(data, title, args.run_dir / "seconds_per_turn.png")
    print(f"wrote {args.run_dir / 'seconds_per_turn.png'} and .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
