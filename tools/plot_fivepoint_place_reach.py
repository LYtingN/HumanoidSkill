"""Plot the fine-tuned five-point policy with demonstration density contours."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from plot_layers import AQUA, BLUE, ORANGE, GRID, SURF, Demo, SceneGeometryCfg, contours

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("PLACE_REACH", "LOWER", "RELEASE")



def read_batches(run):
    parts = []
    for path in sorted(run.glob("batch_*.npz")):
        with np.load(path) as d:
            parts.append(dict(d))
    metadata = ("time", "step", "seed", "checkpoint", "tracker_indices", "tracker_window",
                "action_dim", "noise_repeat", "noise_scale")
    for key in metadata:
        if not all(np.array_equal(d[key], parts[0][key]) for d in parts):
            raise ValueError(f"Batches disagree about {key}")
    data = {k: parts[0][k] for k in metadata}
    for key in ("episodes", "end_time", "outcome", "entered"):
        data[key] = np.concatenate([d[key] for d in parts])
    for key in ("executed", "intent", "follow", "contact", "wrist_poses"):
        data[key] = np.concatenate([d[key] for d in parts], axis=1)
    if len(np.unique(data["episodes"])) != len(data["episodes"]):
        raise ValueError("Repeated episode IDs")
    return data


def recorded_landmarks():
    path = ROOT / "tasks/can_pnp/src/can_pnp/scene/assets/grasp_starts.json"
    doc = json.loads(path.read_text())
    geo = SceneGeometryCfg()
    rim = np.array([*geo.upper_box_center[:2], geo.box_bottom_z + geo.upper_box_size[2]])
    return {phase: {int(i): ((e["phases"][phase]["frame"] - e["phases"]["PLACE_REACH"]["frame"]) / 50.,
                             np.array(e["phases"][phase]["can_pos"]) - rim) for i, e in doc.items()}
            for phase in PHASES}


def verify(d, landmarks):
    a, b, f = (d[k] for k in ("executed", "intent", "follow"))
    scored = np.isfinite(a).all(axis=-1)
    held = scored & d["contact"]
    np.testing.assert_array_equal(np.isfinite(b).all(axis=-1), held)
    np.testing.assert_array_equal(f[scored], a[scored])
    np.testing.assert_allclose(a[0], [landmarks["PLACE_REACH"][i][1] for i in d["episodes"]], atol=5e-6)
    if (scored & (d["time"][:, None] > d["end_time"] + 1e-6)).any():
        raise ValueError("An ended episode contributed to the scored series")
    if (np.diff(np.isfinite(f[..., 0]).astype(int), axis=0) > 0).any():
        raise ValueError("Follow-up includes a trajectory after a reset")
    measured, ref = d["wrist_poses"][held].transpose(1, 0, 2)
    rotate = lambda quat: Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    offset = rotate(measured[:, 3:]).inv().apply(a[held] - measured[:, :3])
    rebuilt = ref[:, :3] + rotate(ref[:, 3:]).apply(offset)
    return float(np.linalg.norm(rebuilt - b[held], axis=-1).max())


def quantiles(xyz):
    count = np.isfinite(xyz[..., 0]).sum(axis=1)
    q = np.full((len(count), 3), np.nan)
    show = count >= 5
    q[show] = np.nanpercentile(np.linalg.norm(xyz[show], axis=-1), [25, 50, 75], axis=1).T
    return count, q


def digitize_demo_distance(path):
    """Read the published green median and IQR from the supplied, calibrated raster."""
    with Image.open(path) as im:
        if im.size != (2080, 676):
            raise ValueError("Expected the supplied 2080 x 676 reference PNG")
        pixels = np.asarray(im.convert("RGB"))
    # Right-panel ticks in pixel centres: 82.5 px/s and 619 px/m.
    t = np.arange(56) / 10
    background = np.array([252, 252, 251])
    colors = np.array([[27, 175, 122], [42, 120, 214], [235, 104, 52]])
    inverse_palette = np.linalg.inv(colors - background)
    q = []
    for seconds in t:
        x = round(1392.2 + 82.5 * seconds)
        patch = pixels[392:519, max(1394, x-1):max(1397, x+2)].astype(float)
        # Undo translucent blue/orange overlays before detecting the green
        # stroke and band. Ignore near-neutral grid and antialiasing pixels.
        weights = (patch - background) @ inverse_palette
        transmission = 1 - weights[..., 1:].clip(0, 1).sum(axis=-1)
        green = weights[..., 0] / transmission.clip(min=.01)
        green[(np.ptp(patch, axis=-1) <= 5) | (weights[..., 0] < .055)] = 0
        band_rows = np.flatnonzero((green > .095).any(axis=1))
        stroke_rows = np.flatnonzero((green > .60).any(axis=1))
        if not len(stroke_rows) or not len(band_rows):
            raise ValueError(f"Green demo unreadable at {seconds:.1f} s")
        rows = np.array([band_rows[-1], np.median(stroke_rows), band_rows[0]]) + 392
        q.append((559.3 - rows) / 619.)
    return t, np.stack(q)


def main(args):
    d = read_batches(args.run)
    landmarks = recorded_landmarks()
    error = verify(d, landmarks)
    if error > 2e-5:
        raise ValueError(f"Wrist projection differs by {error:g} m")
    t = d["time"]
    placed = d["outcome"] == "placed"
    n = len(placed)
    terminal_counts = {str(k): int((d["outcome"] == k).sum()) for k in np.unique(d["outcome"])}
    geo = SceneGeometryCfg()
    w, h = np.array(geo.upper_box_size[:2])/2 - geo.box_wall_thickness
    depth = geo.upper_box_size[2] - geo.box_wall_thickness
    top, side = [[-.6, .4], [-.6, .4]], [[0, .8], [-.15, .35]]
    fig = plt.figure(figsize=(16, 5.2), facecolor=SURF)
    # Match the reference canvas and plotting bounds, including its square top view.
    def panel(left, width):
        return fig.add_axes([left/2080, 117/676, width/2080, 464/676])
    xy, rz, dist = panel(134, 465), panel(747, 528), panel(1392, 660)
    for ax in (xy, rz, dist):
        ax.grid(color=GRID, lw=.6)
        ax.set_axisbelow(True)

    xy.set(xlim=top[0], ylim=top[1], xlabel="x relative to box centre (m)", ylabel="y relative to box centre (m)")
    xy.set_aspect("equal")
    rz.set(xlim=side[0], ylim=side[1], xlabel="horizontal distance to box centre (m)", ylabel="height above the rim (m)")
    dist.set(xlim=(0, 8), ylim=(0, .75), xlabel="time since the PLACE_REACH start (s)",
             ylabel="can distance to the opening centre (m)")
    for ax, title in ((xy, "Top view"), (rz, "Side view"), (dist, "Distance to the box opening (median, 25–75%)")):
        ax.set_title(title, loc="left", fontsize=13, pad=6)

    if args.demo:
        demo_t, demo_xyz = Demo(args.demo).from_phase("PLACE_REACH")
        from plot_layers import band
        band(dist, demo_t, np.linalg.norm(demo_xyz, axis=1), AQUA, 8.)
        demo_source = str(args.demo)
        demo_note = "Demo contours and distance band use recorded trajectories from PLACE_REACH onward."
    else:
        demo_xyz = np.stack([pos for phase in PHASES for _, pos in landmarks[phase].values()])
        demo_t, demo_q = digitize_demo_distance(args.reference)
        dist.fill_between(demo_t, demo_q[:, 0], demo_q[:, 2], color=AQUA, alpha=.15, lw=0)
        dist.plot(demo_t, demo_q[:, 1], color=AQUA, lw=2)
        demo_source = str(args.reference)
        demo_note = (f"Green contours use {len(demo_xyz)} recorded phase starts from {len(landmarks[PHASES[0]])} demos. "
                     "Green median and IQR digitized from the supplied figure.")
    contours(xy, demo_xyz[:, :2], AQUA, top, True)
    contours(rz, np.c_[np.linalg.norm(demo_xyz[:, :2], axis=1), demo_xyz[:, 2]], AQUA, side, True)

    counts, bands = {}, {}
    for key in ("intent", "executed", "follow"):
        points = d[key]
        counts[key], q = quantiles(points)
        if key != "follow":
            color = BLUE if key == "intent" else ORANGE
            samples = points[np.isfinite(points).all(-1)]
            contours(xy, samples[:, :2], color, top, False)
            contours(rz, np.c_[np.linalg.norm(samples[:, :2], axis=1), samples[:, 2]], color, side, False)
        if key != "executed":
            color = BLUE if key == "intent" else ORANGE
            dist.fill_between(t, q[:, 0], q[:, 2], color=color, alpha=.15, lw=0)
            dist.plot(t, q[:, 1], color=color, lw=2)
        bands[key] = [[float(x) for x in row] if np.isfinite(row).all() else None for row in q]

    xy.add_patch(Rectangle((-w, -h), 2*w, 2*h, fill=False, ec="#151515", ls="--", lw=1.5))
    xy.text(0, h+.016, "box opening", ha="center", va="bottom", fontsize=9)
    rz.add_patch(Rectangle((0, -depth), min(w, h), depth, fc=GRID, ec="#52514e", ls="--", lw=1.2, alpha=.5, zorder=0))
    rz.text(min(w, h)/2, -depth-.02, "inside the box", ha="center", va="top", fontsize=9)
    rz.axhline(0, c="#52514e", lw=.8)
    rz.text(.79, .005, "rim height", ha="right", va="bottom", fontsize=8, color="#52514e")
    layers = [(AQUA, "Successful demo · can position"), (BLUE, "Upper layer · reference intent"),
              (ORANGE, "Lower layer · executed can")]
    handles = [Line2D([], [], c=c, lw=2.5, label=label) for c, label in layers]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .999), frameon=False,
               ncol=3, fontsize=10, columnspacing=2, handlelength=2.1)
    for color, label, y in ((BLUE, "upper · reference intent", .43), (ORANGE, "lower · executed can", .37),
                            (AQUA, "successful demo", .05)):
        dist.plot([5.18], [y], marker="s", ms=7.5, color=color)
        dist.text(5.29, y, label, va="center", fontsize=9)
    fig.text(.01, .038,
             f"PLACE_REACH starts. DSRL step {int(d['step']):,}, 47-D repeated noise + MorphMimic. "
             f"{placed.sum()}/{n} placed. Blue uses the FM wrist reference while held. "
             "Orange contours cover scored episodes; the orange distance curve follows the physical can for 8 s.",
             fontsize=7.8, color="#52514e")
    fig.text(.01,.012, demo_note + "  Thick and thin contours enclose 50% and 90% of smoothed density.",
             fontsize=7.8, color="#52514e")
    fig.savefig(args.run/"place_reach_layers_reference_style.png", dpi=130)
    plt.close(fig)
    last = d["follow"][-1]
    in_opening = (np.abs(last[:,0]) < w) & (np.abs(last[:,1]) < h) & (last[:,2] < 0) & (last[:,2] >= -depth)
    stats = dict(checkpoint=str(d["checkpoint"]), step=int(d["step"]), seed=int(d["seed"]), episodes=n,
                 action_dim=int(d["action_dim"]), noise_repeat=bool(d["noise_repeat"]), noise_scale=float(d["noise_scale"]),
                 tracker_window=str(d["tracker_window"]), tracker_frames=d["tracker_indices"].tolist(),
                 outcomes=terminal_counts, placed_median_s=float(np.median(d["end_time"][placed])),
                 max_projection_check_error_m=error, counts={k:v.tolist() for k,v in counts.items()},
                 distance_quartiles_m=bands,
                 follow_at_8s=int(counts["follow"][-1]), can_centre_in_box_at_8s=int(in_opening.sum()),
                 time_s=t.tolist(), demo_phase_starts=len(landmarks["PLACE_REACH"]),
                 demo_contour_observations=len(demo_xyz), demo_contour_source="demo frames" if args.demo else "three phase starts per demo",
                 demo_distance_source=demo_source,
                 full_demo_file=str(args.demo) if args.demo else None)
    (args.run/"summary.json").write_text(json.dumps(stats, indent=2)+"\n")
    print({k: v for k, v in stats.items() if k not in ("counts", "time_s", "distance_quartiles_m")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("reference", type=Path, nargs="?", default=ROOT.parent/"place_reach_layers.png",
                   help="Supplied reference PNG for the published demo distance band")
    p.add_argument("--demo", type=Path, help="Full demo_can_paths.npz, when available")
    main(p.parse_args())
