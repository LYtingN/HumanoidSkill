"""Run a reference source through the SuperMimic tracker in the can pick-and-place scene.

    # Closed loop: retrained diffusion policy (config/can_pnp_supermimic.yaml) -> SuperMimic.
    uv run --extra diffusion can-pnp-run --source diffusion \\
        --diffusion-checkpoint <run>/best.pt --diffusion-repo ../../Diffusion/Humanoid-Diffusion \\
        --tracker-checkpoint <supermimic>/ckpt_0268000.pt

    # The diffusion policy steered by a DSRL noise policy (can-pnp-dsrl checkpoint), from the
    # grasp starts it was trained on.
    uv run --extra diffusion can-pnp-run --source dsrl --noise-checkpoint <dsrl run>/last.pt \\
        --diffusion-checkpoint <run>/best.pt --diffusion-repo ../../Diffusion/Humanoid-Diffusion \\
        --tracker-checkpoint <supermimic>.pt --start grasp --can-mass 0.1 --no-randomize --viewer

    # Open-loop check of tracker + goal construction with recorded targets of one episode.
    uv run can-pnp-run --source dataset --episode <dataset>/data/chunk-000/episode_000000.parquet \\
        --tracker-checkpoint <supermimic>/ckpt_0268000.pt

Reports per env: falls, how far the tracked joints / root heading stay from the reference,
and (closed loop) whether the target can was lifted and placed. Writes head-camera and
overview videos of env 0; ``--viewer`` also shows all envs live in one MuJoCo viewer, with the
head-camera images the diffusion policy sees tiled in the top-right corner.
"""

from __future__ import annotations

import ctypes.util
import faulthandler
import itertools
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Headless CPU rendering through EGL when there is no display but the system has EGL; otherwise
# the overview renders are skipped (the head camera renders on the GPU through mujoco_warp
# either way). With a display, the default GLFW backend also serves the live viewer.
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and ctypes.util.find_library("EGL"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import torch
import tyro
from mjlab.utils.lab_api.math import matrix_from_quat

from can_pnp import mdp
from can_pnp.config import CanPnpSceneCfg, LayoutCfg, SceneGeometryCfg
from can_pnp.constants import BODY_JOINT_NAMES
from can_pnp.dsrl import NoisePolicy, SteeredDiffusionSource
from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.grasp_start import GraspStarts, apply_grasp_start
from can_pnp.layout import layout_term
from can_pnp.pipeline import (
    DatasetSource,
    DiffusionSource,
    StandSource,
    SuperMimicController,
    load_diffusion_policy,
)
from can_pnp.plan_viz import PlanKeypoints
from can_pnp.supermimic import load_tracking_policy


@dataclass
class Args:
    tracker_checkpoint: Path
    source: Literal["diffusion", "dsrl", "dataset", "stand"] = "diffusion"
    diffusion_checkpoint: Path | None = None
    noise_checkpoint: Path | None = None  # dsrl source: a can-pnp-dsrl last.pt / best.pt
    # dsrl source: the policy's mean noise (as evaluated in training), a sample of it, or N(0, I)
    # (the unsteered diffusion policy).
    noise_mode: Literal["mean", "sample", "base"] = "mean"
    diffusion_repo: Path = Path("../../Diffusion/Humanoid-Diffusion")
    episode: Path | None = None  # dataset source: a LeRobot episode parquet
    num_envs: int = 4
    seconds: float = 20.0  # <= 0: no limit, run until the viewer is closed or Ctrl-C
    randomize: bool = True
    # collection: robot starts 0.5-1.5 m from the table and walks over; pregrasp: robot starts
    # standing at the table in the recorded grasp stance (skips the navigation); grasp: every
    # env starts from the recorded state of a collected episode at the start of --grasp-phase
    # (arm already at the can, gripper open; table and cans as recorded).
    start: Literal["collection", "pregrasp", "grasp"] = "collection"
    grasp_phase: Literal["REACH", "CLOSE"] = "CLOSE"
    grasp_episodes: tuple[int, ...] = ()  # LeRobot episode per env (default: env i -> episode i)
    # Mass of both cans (kg); the collected episodes use 0.62. SuperMimic's arm cannot lift that
    # with the gripper's payload, a light can isolates the policy from that limitation.
    can_mass: float | None = None
    device: str = "cuda:0"
    out_dir: Path = Path("logs/pipeline")
    video_envs: int = 1
    seed: int = 0
    viewer: bool = False  # live MuJoCo viewer of all envs (needs a display)
    realtime: bool = True  # viewer: pace the loop to 50 Hz
    show_plan: bool = True  # viewer: future pelvis / hands / feet of the current reference plan


class _Overview:
    """Third-person CPU render of one world following the robot."""

    def __init__(self, env, env_id: int, height: int = 480, width: int = 640):
        self.env, self.env_id = env, env_id
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.0, 215.0, -25.0

    def frame(self) -> np.ndarray:
        i, sim = self.env_id, self.env.sim
        self.data.qpos[:] = sim.data.qpos[i].cpu().numpy()
        self.data.mocap_pos[:] = sim.data.mocap_pos[i].cpu().numpy()
        self.data.mocap_quat[:] = sim.data.mocap_quat[i].cpu().numpy()
        self.model.geom_rgba[:] = sim.model.geom_rgba[i].cpu().numpy()
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.env.scene["robot"].data.root_link_pos_w[i].cpu().numpy()
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()


class _LiveViewer:
    """Passive MuJoCo viewer showing every world of the batch, each at its env origin.

    World 0 is the viewer's own data; the others are added to ``user_scn`` with
    ``mjv_addGeoms`` (as mjlab's native viewer does), each with its own randomized colors.
    Poses are copied from the GPU simulation as computed there: the viewer runs no CPU
    physics (``mj_forward`` on a diverged world can hang the CPU collision code).
    The head-camera images of all envs are tiled in the top-right corner. Enter restarts all
    episodes (as in mjlab's native viewer).
    """

    KEY_ENTER = 257  # GLFW

    # Kinematic outputs of the GPU step that mjv_updateScene / mjv_addGeoms read.
    POSE_FIELDS = (
        "xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos",
        "site_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir", "subtree_com",
    )  # fmt: skip

    def __init__(self, env, tile: tuple[int, int] = (160, 120), columns: int = 4):
        import mujoco.viewer

        self.env, self.tile, self.columns = env, tile, columns
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.other = mujoco.MjData(self.model)
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        self._last: dict[str, np.ndarray] | None = None
        self.reset_requested = False
        self.handle = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=self._key
        )
        origins = env.scene.env_origins.cpu().numpy()
        span = float(np.ptp(origins[:, :2], axis=0).max()) if len(origins) > 1 else 0.0
        with self.handle.lock():
            self.handle.cam.lookat[:] = origins.mean(0) + np.array([0.5, 0.0, 0.6])
            self.handle.cam.distance = 4.0 + 1.2 * span
            self.handle.cam.azimuth, self.handle.cam.elevation = 215.0, -35.0

    def _key(self, key: int) -> None:
        # Viewer thread: only flag the request, the main loop resets the simulation.
        if key == self.KEY_ENTER:
            self.reset_requested = True

    @property
    def running(self) -> bool:
        return self.handle.is_running()

    def _set_world(self, data: mujoco.MjData, state: dict[str, np.ndarray], i: int) -> None:
        for key, value in state.items():
            if key == "geom_rgba":
                self.model.geom_rgba[:] = value[i]
            else:
                dst = getattr(data, key)
                dst[:] = value[i].reshape(dst.shape)
        data.ncon = 0

    def update(self, heads: np.ndarray | None, text: str, draw=None) -> None:
        sim = self.env.sim
        state = {k: getattr(sim.data, k).cpu().numpy() for k in ("qpos", "qvel", "mocap_pos", "mocap_quat", *self.POSE_FIELDS)}
        state["geom_rgba"] = sim.model.geom_rgba.cpu().numpy()
        # A diverged world is drawn where it was last finite.
        bad = ~np.isfinite(state["geom_xpos"]).all(axis=(1, 2))
        if bad.any() and self._last is not None:
            for key, value in state.items():
                value[bad] = self._last[key][bad]
        self._last = state
        with self.handle.lock():
            scn = self.handle.user_scn
            scn.ngeom = 0
            for i in range(1, self.env.num_envs):
                self._set_world(self.other, state, i)
                mujoco.mjv_addGeoms(
                    self.model, self.other, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn
                )
            self._set_world(self.data, state, 0)
            if draw is not None:
                draw(scn)
        # Outside the lock: set_texts / set_images wait for the render thread to take the
        # overlay, and the render thread needs the lock to do so (deadlock otherwise).
        self.handle.set_texts((None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, text, ""))
        if heads is not None:
            w, h = self.tile
            step = max(1, heads.shape[2] // w)
            small = heads[:, ::step, ::step][:, :h, :w]
            rows = -(-len(small) // self.columns)
            grid = np.zeros((rows * h, self.columns * w, 3), dtype=np.uint8)
            for k, image in enumerate(small):
                r, c = divmod(k, self.columns)
                grid[r * h : r * h + image.shape[0], c * w : c * w + image.shape[1]] = image
            port = self.handle.viewport
            if port is not None and port.width > grid.shape[1] and port.height > grid.shape[0]:
                rect = mujoco.MjrRect(port.width - grid.shape[1], port.height - grid.shape[0], grid.shape[1], grid.shape[0])
                self.handle.set_images((rect, grid))
        self.handle.sync()

    def close(self) -> None:
        self.handle.close()


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    # `kill -USR1 <pid>` prints where every thread is, without stopping the run.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    closed_loop = args.source in ("diffusion", "dsrl")
    need_camera = closed_loop or args.video_envs > 0 or args.viewer
    unlimited = args.seconds <= 0
    env = make_env(
        CanPnpEnvCfgArgs(
            num_envs=args.num_envs, randomize=args.randomize, camera=need_camera,
            episode_length_s=1e9 if unlimited else args.seconds + 1.0, seed=args.seed,
            scene=CanPnpSceneCfg(
                layout=LayoutCfg(robot_start="pregrasp" if args.start == "pregrasp" else "collection"),
                geometry=SceneGeometryCfg(**({} if args.can_mass is None else {"can_mass": args.can_mass})),
            ),
        ),
        device=args.device,
    )  # fmt: skip
    tracker = load_tracking_policy(args.tracker_checkpoint, device=args.device)
    robot = env.scene["robot"]
    if closed_loop:
        if args.diffusion_checkpoint is None:
            raise SystemExit(f"--diffusion-checkpoint is required for --source {args.source}")
        policy, _ = load_diffusion_policy(args.diffusion_checkpoint, args.diffusion_repo, args.device)
        source = DiffusionSource(policy)
    if args.source == "dsrl":
        if args.noise_checkpoint is None:
            raise SystemExit("--noise-checkpoint is required for --source dsrl")
        state = torch.load(args.noise_checkpoint, map_location=args.device, weights_only=False)
        noise_policy = NoisePolicy(policy.n_tokens, policy.n_emb, policy.horizon, policy.action_dim,
                                   hidden=state["args"]["hidden"]).to(args.device)  # fmt: skip
        noise_policy.load_state_dict(state["noise_policy"])
        source = SteeredDiffusionSource(policy, noise_policy.eval(), args.noise_mode, record=False)
        print(f"dsrl: {args.noise_checkpoint} (iteration {state['iteration']}), noise {args.noise_mode}")
    elif args.source == "dataset":
        if args.episode is None:
            raise SystemExit("--episode is required for --source dataset")
        source = DatasetSource(args.episode, args.num_envs, args.device)
    else:
        body_ids = robot.find_joints(BODY_JOINT_NAMES, preserve_order=True)[0]
        source = StandSource(robot.data.default_joint_pos[:, body_ids])
    controller = SuperMimicController(
        env, source, tracker, camera="head_camera" if closed_loop else None
    )

    obs, _ = env.reset()
    grasp = None
    if args.start == "grasp":
        grasp = GraspStarts(phase=args.grasp_phase)
        episodes = args.grasp_episodes or tuple(range(env.num_envs))
        if len(episodes) != env.num_envs:
            raise SystemExit(f"--grasp-episodes needs {env.num_envs} episodes, got {len(episodes)}")
        grasp_rows = torch.tensor([grasp.episode_ids.index(e) for e in episodes])
        apply_grasp_start(env, torch.arange(env.num_envs, device=env.device), grasp, grasp_rows)
        obs = env.obs_buf
        print(f"grasp start: episodes {list(episodes)} at {args.grasp_phase}")
    n, device = env.num_envs, env.device
    reset = torch.ones(n, dtype=torch.bool, device=device)
    steps = None if unlimited else int(round(args.seconds / env.step_dt))
    geo = env.cfg.events["reset_layout"].params["scene_cfg"].geometry
    can = env.scene["can"]
    fallen = torch.zeros(n, dtype=torch.bool, device=device)
    lifted = torch.zeros(n, dtype=torch.bool, device=device)
    placed = torch.zeros(n, dtype=torch.bool, device=device)
    joint_err = torch.zeros(n, device=device)
    heading_err = torch.zeros(n, device=device)
    counted = torch.zeros(n, device=device)
    videos = {"head": [], "overview": []}
    overview = None
    if args.video_envs > 0:
        try:
            overview = _Overview(env, 0)
        except (AttributeError, ImportError, mujoco.FatalError) as error:
            # The CPU renderer needs an EGL/OSMesa library; the head camera renders on the GPU.
            print(f"overview video skipped: no OpenGL context for MUJOCO_GL ({error})")
    viewer = _LiveViewer(env) if args.viewer else None
    plan_points = PlanKeypoints(env, controller) if args.viewer and args.show_plan else None
    step, start_step = -1, 0
    try:
        for step in itertools.count() if steps is None else range(steps):
            tic = time.perf_counter()
            if viewer is not None and viewer.reset_requested:
                viewer.reset_requested = False
                obs, _ = env.reset()
                if grasp is not None:
                    apply_grasp_start(env, torch.arange(n, device=device), grasp, grasp_rows)
                    obs = env.obs_buf
                reset = torch.ones(n, dtype=torch.bool, device=device)
                for buf in (fallen, lifted, placed, joint_err, heading_err, counted):
                    buf.zero_()
                start_step = step
                print(f"viewer reset at step {step}")
            action = controller.act(obs, reset)
            if plan_points is not None:
                plan_points.update()
            # Tracking error against the reference frame being tracked this step.
            plan, offset = controller.plan, controller.step_in_chunk - 1
            rows = torch.arange(n, device=device)
            ref_q = plan.joint_pos[rows, offset]
            joint_err += (robot.data.joint_pos[:, controller.body_ids] - ref_q).abs().mean(-1) * ~fallen
            ref_fwd = plan.root_w[rows, offset][:, :2, 0]
            fwd = matrix_from_quat(robot.data.root_link_quat_w)[:, :2, 0]
            cos = torch.nn.functional.cosine_similarity(fwd, ref_fwd, dim=-1).clamp(-1, 1)
            heading_err += torch.rad2deg(torch.acos(cos)) * ~fallen
            counted += ~fallen

            obs, _, terminated, truncated, _ = env.step(action)
            reset = terminated | truncated
            if grasp is not None and bool(reset.any()):
                ids = reset.nonzero().flatten()
                apply_grasp_start(env, ids, grasp, grasp_rows[ids.cpu()])
                obs = env.obs_buf
            # Episodes are as long as the run, so a reset here is a fall or a dropped can.
            fallen |= mdp.robot_fallen(env) | (terminated & ~fallen)
            can_z = can.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            lifted |= can_z > geo.table_height + geo.can_height / 2 + 0.05
            box = layout_term(env).box_center_w
            inner = torch.tensor(geo.upper_box_size[:2], device=device) / 2 - geo.box_wall_thickness
            rel = (can.data.root_link_pos_w[:, :2] - box[:, :2]).abs()
            placed |= (rel < inner).all(-1) & (can_z < geo.box_bottom_z + geo.can_height)
            if overview is not None:
                videos["overview"].append(overview.frame())
            if args.video_envs > 0 and "head_camera" in env.scene.sensors:
                videos["head"].append(env.scene["head_camera"].data.rgb[0].cpu().numpy())
            if viewer is not None:
                if not viewer.running:
                    print(f"viewer closed at step {step}")
                    break
                heads = None
                if "head_camera" in env.scene.sensors:
                    heads = env.scene["head_camera"].data.rgb.cpu().numpy()
                finite = torch.isfinite(env.sim.data.qpos).all(-1).cpu()
                status = "\n".join(
                    f"env {i}: lifted {int(lifted[i])}  in box {int(placed[i])}  fallen {int(fallen[i])}"
                    + ("" if finite[i] else "  NaN")
                    for i in range(n)
                )
                legend = f"\n{PlanKeypoints.legend()}" if plan_points is not None else ""
                viewer.update(heads, f"t {(step + 1 - start_step) * env.step_dt:5.2f} s   source {args.source}  (Enter: reset)\n{status}{legend}",
                              draw=plan_points.draw if plan_points is not None else None)
                if args.realtime:
                    time.sleep(max(0.0, env.step_dt - (time.perf_counter() - tic)))
    except KeyboardInterrupt:
        print(f"interrupted at step {step}")
    if viewer is not None:
        viewer.close()

    counted = counted.clamp_min(1)
    steps = step + 1
    print(f"source={args.source} envs={n} steps={steps} ({steps * env.step_dt:.1f} s)")
    print(f"  fallen        {fallen.cpu().tolist()}")
    print(f"  joint |err|   {(joint_err / counted).cpu().numpy().round(4).tolist()} rad (mean over tracked steps)")
    print(f"  heading err   {(heading_err / counted).cpu().numpy().round(2).tolist()} deg")
    if closed_loop:
        print(f"  can lifted    {lifted.cpu().tolist()}")
        print(f"  can in box    {placed.cpu().tolist()}")
    if any(videos.values()):
        import mediapy

        fps = int(round(1.0 / env.step_dt))
        for name, frames in videos.items():
            if frames:
                path = args.out_dir / f"{args.source}_{name}_env0.mp4"
                mediapy.write_video(path, frames, fps=fps)
                print(f"  wrote {path}")


if __name__ == "__main__":
    main()
