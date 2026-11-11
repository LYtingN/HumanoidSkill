"""Local mjlab closed loop: five-point FM -> MorphMimic -> G1 + ALOHA.

Run from the workspace with humanoid_skill/.venv/bin/python -m
experiments.view_fivepoints. Videos and numerical diagnostics are always saved.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# can_pnp imports mujoco before mjlab sets MUJOCO_GL=egl; without it the offscreen renders fall back
# to GLFW, i.e. Mesa llvmpipe on this X display (~180 ms per overview frame instead of ~1 ms).
os.environ.setdefault("MUJOCO_GL", "egl")
# The X server runs the NVIDIA driver, but glvnd hands GLX windows (the viewer) to Mesa llvmpipe.
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import numpy as np
import torch
import tyro

WORKSPACE = Path(__file__).resolve().parents[2]


@dataclass
class Args:
    fm_checkpoint: Path = WORKSPACE / "downloads/5points_FM/fivepoints_fm.pt"
    tracker_checkpoint: Path = WORKSPACE / "downloads/morphmimic/ckpt_45000.pt"
    fm_source: Path = WORKSPACE / "downloads/5points_FM/inference_source"
    morphmimic_source: Path = WORKSPACE / "downloads/morphmimic/inference_source"
    out_dir: Path = WORKSPACE / "outputs/fivepoints_mjlab/ckpt45000_close"
    seconds: float = 12.0
    device: str = "cuda:0"
    seed: int = 42
    # Negative: collection layout. Nonnegative: recorded grasp start of that episode.
    episode: int = 3
    phase: Literal["REACH", "CLOSE"] = "CLOSE"
    can_mass: float = 0.05  # kg
    # mjlab's native viewer (realtime by its sim-time budget); no videos or trace in this mode.
    viewer: bool = False
    viewer_stats: bool = False  # print the viewer's FPS and real-time factor every 0.5 s
    viewer_fps: float = 30.0
    # The viewer opens at 2/3 of the 5120 x 2824 screen; presenting frames that large on this
    # virtual display limits it to ~0.08x real time (1280 x 800: ~0.5x).
    viewer_size: tuple[int, int] = (1280, 800)


# Pelvis yellow, left / right ankle blue / green, left / right wrist red / purple.
PLAN_COLORS = [(1., .8, .1, .8), (.2, .6, 1., .8), (.2, 1., .4, .8),
               (1., .3, .3, .8), (.8, .3, 1., .8)]


def draw_plan(scene, plan, offset):
    import mujoco

    points = plan.pos_w[0, offset:].cpu().numpy()
    for k, color in enumerate(PLAN_COLORS):
        for p in points[::3, k]:
            if scene.ngeom >= scene.maxgeom:
                return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                               np.array([.009] * 3), p, np.eye(3).ravel(), np.array(color))
            scene.ngeom += 1


class PlanMarkers:
    """The FM's remaining world-frame keypoint plan as debug spheres of mjlab's viewer (registered
    in ``env.manager_visualizers``, drawn every viewer frame)."""

    def __init__(self, controller):
        self.controller = controller

    def debug_vis(self, visualizer) -> None:
        plan = self.controller.plan
        if plan is None:
            return
        offset = max(int(self.controller.step_in_chunk[0]) - 1, 0)
        points = plan.pos_w[0, offset::3].cpu().numpy()
        for k, color in enumerate(PLAN_COLORS):
            for p in points[:, k]:
                visualizer.add_sphere(p, 0.009, color)


class ViewerPolicy:
    """The controller as an mjlab viewer policy. A terminated episode (the env resets it to the
    collection layout) and the viewer's reset (Enter) restart from the recorded grasp start."""

    def __init__(self, env, controller, restart, window_size: tuple[int, int]):
        self.env, self.controller, self.restart = env, controller, restart
        self.reset_mask = torch.ones(1, dtype=torch.bool, device=env.device)
        self.window_size = window_size  # applied once the viewer's window exists

    def __call__(self, obs):
        from experiments.view_policy import resize_window

        if self.window_size is not None and resize_window("MuJoCo : mjlab scene", *self.window_size):
            self.window_size = None
        # env.obs_buf, not the viewer's obs: get_observations() does not advance the history.
        action = self.controller.act(self.env.obs_buf, self.reset_mask)
        self.reset_mask = torch.zeros_like(self.reset_mask)
        return action

    def reset(self, dones=None):
        if dones is None or bool(dones.any()):
            self.restart()
            self.reset_mask = torch.ones_like(self.reset_mask)


def main(args: Args | None = None):
    args = args or tyro.cli(Args)
    sys.path[:0] = [str(args.fm_source), str(args.morphmimic_source)]
    from can_pnp.scene.config import CanPnpSceneCfg, SceneGeometryCfg
    from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
    from can_pnp.scene.layout import layout_term
    from can_pnp.starts import GraspStarts, apply_grasp_start
    from experiments.view_policy import _Overview
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.viewer.base import VerbosityLevel
    from hskill.s0.morphmimic import KeypointTracker, MorphMimicController
    from humanoid_diffusion.model import load_policy
    import imageio.v2 as imageio

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("GPU", torch.cuda.get_device_name(args.device), "free,total", torch.cuda.mem_get_info(), flush=True)
    fm, config = load_policy(args.fm_checkpoint, device=args.device)
    tracker = KeypointTracker(args.tracker_checkpoint, args.device)
    env = make_env(CanPnpEnvCfgArgs(num_envs=1, randomize=False, camera=True,
                                   episode_length_s=max(args.seconds + 1, 1e9), seed=args.seed,
                                   scene=CanPnpSceneCfg(geometry=SceneGeometryCfg(can_mass=args.can_mass))),
                   args.device)
    print(f"Can mass {args.can_mass * 1000:.0f} g", flush=True)
    overview = None
    writers = []
    records = []
    failure = None
    starts = GraspStarts(phase=args.phase) if args.episode >= 0 else None

    def restart():
        env.reset()
        if starts is not None:
            start_row = starts.episode_ids.index(args.episode)
            apply_grasp_start(env, torch.tensor([0], device=args.device), starts,
                              torch.tensor([start_row]))
            print(f"Start: episode {args.episode}, {args.phase}, recorded frame "
                  f"{int(starts.frame[start_row])}", flush=True)
        return env.obs_buf

    try:
        obs = restart()
        controller = MorphMimicController(env, fm, tracker)
        if args.viewer:
            print("mjlab native viewer: Enter restarts from the start, Space pauses", flush=True)
            verbosity = VerbosityLevel.DEBUG if args.viewer_stats else VerbosityLevel.SILENT
            env.manager_visualizers["fm_plan"] = PlanMarkers(controller)
            policy = ViewerPolicy(env, controller, restart, args.viewer_size)
            NativeMujocoViewer(env, policy, frame_rate=args.viewer_fps, verbosity=verbosity).run()
            return
        overview = _Overview(env, 0, height=480, width=640)
        for name in ("overview", "head"):
            writers.append(imageio.get_writer(str(args.out_dir / f"{name}.mp4"), fps=50,
                                              codec="libx264", quality=8))
        reset = torch.ones(1, dtype=torch.bool, device=args.device)
        total = int(round(args.seconds / env.step_dt)) if args.seconds > 0 else None
        i, wall_start = 0, time.perf_counter()
        print("Running FM(32 x 47, replan every 8) -> MorphMimic(10 x 5 poses) at 50 Hz", flush=True)
        while total is None or i < total:
            tick = time.perf_counter()
            action = controller.act(obs, reset)
            offset = int(controller.step_in_chunk[0]) - 1
            target = controller.plan.pos_w[0, offset]
            measured = controller.robot.data.body_link_pos_w[0, controller.kp_ids]
            # World-frame error, BEFORE physics, against the current target frame.
            error = (measured - target).norm(dim=-1).cpu().tolist()
            draw = lambda scene: draw_plan(scene, controller.plan, offset)
            frame = overview.frame(draw)
            head = env.scene["head_camera"].data.rgb[0].cpu().numpy()
            writers[0].append_data(frame)
            writers[1].append_data(head)
            if i == 0:
                imageio.imwrite(args.out_dir / "overview_initial.png", frame)
                imageio.imwrite(args.out_dir / "head_initial.png", head)
            obs, _, terminated, truncated, _ = env.step(action)
            fallen = bool(env.termination_manager.get_term("fallen")[0])
            dropped = bool(env.termination_manager.get_term("can_dropped")[0])
            finite = bool(torch.isfinite(env.sim.data.qpos).all())
            geo = layout_term(env).scene_cfg.geometry
            can = env.scene["can"].data.root_link_pos_w[0]
            box = layout_term(env).box_center_w[0]
            inner = torch.tensor(geo.upper_box_size[:2], device=args.device) / 2 - geo.box_wall_thickness
            records.append(dict(step=i, time_s=i * .02, kp_error_m=error,
                                pelvis_z=float(controller.robot.data.root_link_pos_w[0, 2]),
                                can_z=float(can[2]),
                                lifted=bool(can[2] > geo.table_height + geo.can_height / 2 + .05),
                                in_box=bool(((can[:2] - box[:2]).abs() < inner).all()
                                            & (can[2] < geo.box_bottom_z + geo.can_height)),
                                fallen=fallen, dropped=dropped, finite=finite,
                                action_max=float(action[:, :29].abs().max()),
                                wall_step_s=time.perf_counter() - tick))
            reset = terminated | truncated
            i += 1
            if i % 50 == 0:
                print(f"t={i*.02:.1f}s pelvis_z={records[-1]['pelvis_z']:.3f} "
                      f"keypoint_mm={np.round(np.array(error)*1000, 1).tolist()}", flush=True)
            if bool(reset.any()) or not finite:
                failure = "fallen" if fallen else "can_dropped" if dropped else "nonfinite" if not finite else "terminated"
                print(f"Stopped on {failure}; no hidden episode restart", flush=True)
                break
        elapsed = time.perf_counter() - wall_start
        summary = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       device=torch.cuda.get_device_name(args.device), steps=len(records),
                       simulation_seconds=len(records)*.02, wall_seconds=elapsed,
                       replans=controller.replans, stopped_reason=failure,
                       lifted=any(r["lifted"] for r in records), in_box=any(r["in_box"] for r in records),
                       kp_mean_error_mm=(np.mean([r["kp_error_m"] for r in records], axis=0)*1000).tolist()
                       if records else [],
                       limitations=["Offline-selected checkpoint; this is a closed-loop diagnostic, not a success-rate evaluation.",
                                    "Uses existing task G1+ALOHA with hand gravity compensation enabled.",
                                    "Rendering, friction and gripper dynamics may differ from collection; synchronous inference excludes hardware latency."])
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.out_dir / "trace.json").write_text(json.dumps(records) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for writer in writers:
            writer.close()
        if overview is not None:
            overview.renderer.close()
        env.close()


if __name__ == "__main__":
    main()
