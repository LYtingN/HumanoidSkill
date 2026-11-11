"""Where the can must go, where S1 sends it and where S0 takes it, one frame per collection of a run.

    .venv/bin/python tools/plot_layers.py logs/v3/<run>               # frames + <run>/layers.mp4
    .venv/bin/python tools/plot_layers.py logs/v3/<run> --watch 120   # follow the run

``train_v3`` writes ``<run>/layers/<cycle>.npz`` per collection (hskill.utils.distributions.layer_points:
the frontier episodes of rank 0, every 5 steps). Each becomes ``<run>/layers/frames/<cycle>.png``:

- green, the demonstrations (``logs/demo_can_paths.npz``, tools/extract_demo_can.py) from the same
  phase on: where the can must go;
- blue, S1's intent: the can carried by the reference executed exactly (while held);
- orange, S0's execution: the can as simulated;

as a top view and a side view of the box (contours enclosing 50 % and 90 % of the samples) and the
distance to the box opening over the episode (median, 25-75 %). The frames make ``<run>/layers.mp4``;
``<run>/layers_latest.png`` is the newest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks/can_pnp/src"))
from can_pnp.scene.config import SceneGeometryCfg  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = (("Demo (successful): where the can must go", AQUA), ("S1 (upper layer): intent", BLUE),
          ("S0 (lower layer): executed can", ORANGE))  # fmt: skip
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})  # fmt: skip


class Demo:
    def __init__(self, path: Path):
        d = np.load(path)
        self.rows, self.start, self.phases = d["rows"], d["phase_start"], [str(p) for p in d["phases"]]

    def from_phase(self, phase: str) -> tuple[np.ndarray, np.ndarray]:
        """(t since the phase start, xyz) of every frame from ``phase`` on, of the episodes reaching it."""
        start = self.start[:, self.phases.index(phase)][self.rows[:, 0].astype(int)]
        keep = np.isfinite(start) & (self.rows[:, 1] >= start)
        return self.rows[keep, 1] - start[keep], self.rows[keep, 3:6]


def contours(ax, xy: np.ndarray, color: str, ext, filled: bool) -> None:
    """Contours enclosing 50 % (thick) and 90 % (thin) of the points (smoothed 2-D histogram)."""
    if len(xy) < 20:
        return
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=120, range=ext)
    H = gaussian_filter(H, 2.0).T
    v = np.sort(H.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    levels = sorted({float(v[min(np.searchsorted(c, q), len(v) - 1)]) for q in (0.9, 0.5)})
    levels = [lv for lv in levels if lv > 0]
    if not levels:
        return
    X, Y = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2)
    if filled:
        ax.contourf(X, Y, H, levels=levels + [H.max()], colors=[color], alpha=0.18)
    ax.contour(X, Y, H, levels=levels, colors=[color], linewidths=[1.2, 2.0][: len(levels)])


def band(ax, t: np.ndarray, dist: np.ndarray, color: str, t_max: float) -> None:
    tb = np.round(t / 0.1) * 0.1
    ts = np.unique(tb)
    ts = ts[ts <= t_max]
    q = np.array([np.percentile(dist[tb == x], [25, 50, 75]) if (tb == x).sum() >= 5 else [np.nan] * 3 for x in ts])
    if len(q):
        ax.fill_between(ts, q[:, 0], q[:, 2], color=color, alpha=0.15, lw=0)
        ax.plot(ts, q[:, 1], color=color, lw=2)


def extents(demo_xyz: np.ndarray) -> tuple[list, list]:
    """Top-view and side-view ranges around the phase's demonstrations (fixed per phase)."""
    lo, hi = np.percentile(demo_xyz, 1, axis=0), np.percentile(demo_xyz, 99, axis=0)
    centre, half = (lo[:2] + hi[:2]) / 2, max(float((hi[:2] - lo[:2]).max()) / 2 + 0.2, 0.4)
    top = [[centre[0] - half, centre[0] + half], [centre[1] - half, centre[1] + half]]
    r_max = max(float(np.linalg.norm(demo_xyz[:, :2], axis=1).max()) + 0.2, 0.6)
    side = [[0.0, r_max], [min(float(lo[2]) - 0.05, -0.15), max(float(hi[2]) + 0.15, 0.3)]]
    return top, side


def frame(npz: Path, demo: Demo, geo: SceneGeometryCfg, out: Path) -> None:
    d = np.load(npz)
    phase, cycle, t_max = str(d["phase"]), int(d["cycle"]), float(d["episode_s"])
    demo_t, demo_xyz = demo.from_phase(phase)
    data = ((demo_t, demo_xyz), (d["intent_t"].astype(np.float32), d["intent"].astype(np.float32)),
            (d["executed_t"].astype(np.float32), d["executed"].astype(np.float32)))  # fmt: skip
    top, side = extents(demo_xyz)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    ax = axes[0]
    for (_, color), (_, xyz) in zip(SERIES, data):
        contours(ax, xyz[:, :2], color, top, filled=color == AQUA)
    w = geo.upper_box_size[0] / 2 - geo.box_wall_thickness
    h = geo.upper_box_size[1] / 2 - geo.box_wall_thickness
    ax.add_patch(Rectangle((-w, -h), 2 * w, 2 * h, fill=False, ec=INK, lw=1.5, ls="--"))
    ax.text(0, h + 0.015, "box opening", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xlim(*top[0])
    ax.set_ylim(*top[1])
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("x relative to box centre (m)")
    ax.set_ylabel("y relative to box centre (m)")
    ax.set_title("Top view", color=INK, loc="left")

    ax = axes[1]
    for (_, color), (_, xyz) in zip(SERIES, data):
        contours(ax, np.c_[np.linalg.norm(xyz[:, :2], axis=1), xyz[:, 2]], color, side, filled=color == AQUA)
    depth = geo.upper_box_size[2] - geo.box_wall_thickness
    ax.add_patch(Rectangle((0, -depth), min(w, h), depth, fc=GRID, ec=INK, lw=1.2, ls="--", alpha=0.6))
    ax.text(min(w, h) / 2, -depth - 0.02, "inside the box", ha="center", va="top", color=INK, fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.text(side[0][1] - 0.01, 0.005, "rim height", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.set_xlim(*side[0])
    ax.set_ylim(*side[1])
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel("horizontal distance to box centre (m)")
    ax.set_ylabel("height above the rim (m)")
    ax.set_title("Side view", color=INK, loc="left")

    ax = axes[2]
    for (_, color), (t, xyz) in zip(SERIES, data):
        band(ax, t, np.linalg.norm(xyz, axis=1), color, t_max)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, max(0.75, float(np.percentile(np.linalg.norm(demo_xyz, axis=1), 99)) + 0.1))
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel(f"time since the {phase} start (s)")
    ax.set_ylabel("can distance to the opening centre (m)")
    ax.set_title("Distance to the box opening (median, 25-75%)", color=INK, loc="left")

    handles = [plt.Line2D([], [], color=c, lw=2.5, label=n) for n, c in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False, fontsize=10,
               labelcolor=INK)  # fmt: skip
    placed, failed, timeout = d["outcomes"] / max(float(d["outcomes"].sum()), 1)
    fig.text(0.01, 0.985, f"cycle {cycle}   frontier {phase}   placed {placed:.0%}   failed {failed:.0%}   "
             f"timed out {timeout:.0%}", fontsize=12, color=INK, va="top")  # fmt: skip
    fig.text(0.01, 0.005, f"{phase} starts. Sim: the frontier episodes of rank 0 ({len(data[2][0])} can samples, "
             f"every 0.1 s); intent = the can carried by the full-body reference executed exactly, while held.\n"
             f"Demo: {int(np.isfinite(demo.start[:, demo.phases.index(phase)]).sum())} collected episodes from "
             f"{phase} on. Contours enclose 50% (thick) and 90% (thin) of the samples.", fontsize=8, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def update(run: Path, demo: Demo, geo: SceneGeometryCfg, fps: float) -> None:
    frames_dir = run / "layers" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    new = 0
    for npz in sorted((run / "layers").glob("*.npz")):
        png = frames_dir / f"{npz.stem}.png"
        if png.exists():
            continue
        try:
            frame(npz, demo, geo, png)
        except (OSError, ValueError, KeyError) as error:  # an npz still being written
            print(f"{npz.name}: skipped ({error})")
            continue
        new += 1
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs or not new:
        print(f"{run}: {len(pngs)} frames, nothing new")
        return
    import imageio_ffmpeg
    import mediapy

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    images = [mediapy.read_image(p)[..., :3] for p in pngs]
    h, w = images[0].shape[:2]
    images = [im[: h - h % 2, : w - w % 2] for im in images if im.shape[:2] == (h, w)]
    mediapy.write_video(run / "layers.mp4", images, fps=fps)
    (run / "layers_latest.png").write_bytes(pngs[-1].read_bytes())
    print(f"{run}: {new} new frames, {len(images)} in layers.mp4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--demo", type=Path, default=ROOT / "logs" / "demo_can_paths.npz")
    parser.add_argument("--fps", type=float, default=4.0, help="collections per second of video")
    parser.add_argument("--watch", type=float, default=0.0, help="update every this many seconds (0: once)")
    args = parser.parse_args()
    demo, geo = Demo(args.demo), SceneGeometryCfg()
    while True:
        update(args.run, demo, geo, args.fps)
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
