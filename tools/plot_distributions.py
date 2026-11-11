"""Draw a train_v3 run's distributions over its cycles (``<run>/dists.jsonl``, see hskill.utils.distributions).

    .venv/bin/python tools/plot_distributions.py logs/v3/<run>              # -> <run>/distributions.png
    .venv/bin/python tools/plot_distributions.py logs/v3/<run> --watch 120  # redraw every 2 min

Top: the share of the frontier's episodes placed / failed / timed out. Below, one heatmap per
distribution: x the cycle, y the value, colour the share of that cycle's samples in the bin (each
cycle normalized on its own), with the median drawn over it. Dashed lines mark curriculum advances.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PANELS = (
    ("success_potential", "timed-out episodes: final task potential"),
    ("s1_noise_norm", "S1 noise  |w| / sqrt(dim)   (N(0, I): 1)"),
    ("s1_intent_hand_can_cm", "S1 intent: reference right hand to can (cm)"),
    ("s0_hand_err_cm", "S0 execution: right hand to reference (cm)"),
)
OUTCOME_COLORS = {"placed": "#1baf7a", "failed": "#e34948", "timeout": "#c3c2b7"}
# Sequential, one hue light -> dark (blue ramp 100 .. 700).
DENSITY = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#0b0b0b", "#52514e"


def load(run: Path) -> list[dict]:
    rows = []
    for line in (run / "dists.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # the line being written
    return rows


def median(edges: np.ndarray, counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    k = int(np.searchsorted(np.cumsum(counts), total / 2))
    return float(0.5 * (edges[k] + edges[k + 1]))


def draw(run: Path, out: Path) -> None:
    rows = load(run)
    if not rows:
        print(f"{run}: no dists.jsonl records yet")
        return
    cycles = np.array([r["cycle"] for r in rows])
    levels = np.array([r["level"] for r in rows])
    advances = cycles[1:][np.diff(levels) > 0]
    fig, axes = plt.subplots(1 + len(PANELS), 1, figsize=(11, 2.2 * (1 + len(PANELS))), sharex=True,
                             gridspec_kw={"height_ratios": [1.0] + [1.3] * len(PANELS)})  # fmt: skip

    ax = axes[0]
    counts = np.array([r["success"]["counts"] for r in rows])
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    ax.stackplot(cycles, share.T, colors=[OUTCOME_COLORS[o] for o in rows[0]["outcomes"]],
                 labels=rows[0]["outcomes"], edgecolor="#fcfcfb", linewidth=0.5)  # fmt: skip
    ax.set_ylim(0, 1)
    ax.set_title("frontier episodes: outcome share", fontsize=10, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    # An invisible colorbar keeps this panel's x-axis aligned with the heatmaps'.
    fig.colorbar(plt.cm.ScalarMappable(cmap=DENSITY), ax=ax, pad=0.01, fraction=0.025).ax.set_visible(False)

    for ax, (name, title) in zip(axes[1:], PANELS):
        edges = np.array(rows[0][name]["edges"])
        hist = np.array([r[name]["counts"] for r in rows])  # (cycles, bins)
        density = hist / np.maximum(hist.sum(1, keepdims=True), 1)
        # Each cycle spans from its own cycle number to the next one's.
        x = np.append(cycles, cycles[-1] + (cycles[-1] - cycles[-2] if len(cycles) > 1 else 1))
        mesh = ax.pcolormesh(x, edges, density.T, cmap=DENSITY, shading="flat", vmin=0)
        ax.plot(cycles, [median(edges, h) for h in hist], color="#eb6834", lw=1.2, label="median")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025).set_label("share of cycle", fontsize=8, color=MUTED)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    for ax in axes:
        for c in advances:
            ax.axvline(c, color=MUTED, lw=0.8, ls="--")
        ax.tick_params(labelsize=8, colors=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
    for c, lvl in zip(advances, levels[1:][np.diff(levels) > 0]):
        axes[0].text(c, 1.02, f"level {lvl}", fontsize=7, color=MUTED, ha="center", va="bottom",
                     transform=axes[0].get_xaxis_transform())  # fmt: skip
    axes[-1].set_xlabel("cycle", fontsize=9, color=MUTED)
    fig.suptitle(run.name, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"{out}: {len(rows)} collections, cycles {cycles[0]} .. {cycles[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--watch", type=float, default=0.0, help="redraw every this many seconds (0: once)")
    args = parser.parse_args()
    out = args.out or args.run / "distributions.png"
    while True:
        draw(args.run, out)
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
