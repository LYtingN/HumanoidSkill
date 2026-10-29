"""Build the can pick-and-place scene, verify its layout and render it.

    MUJOCO_GL=egl uv run can-pnp-check --num-envs 8 --out-dir logs/scene_check

Checks after reset (per env): target can resting on the tabletop, resident can on the box
floor and inside the box, robot clear of the table. Then holds the default pose with zero
actions and reports how far the cans move while the props stay put. Writes the head-camera
image of every env, an overview render of the first envs and the compiled scene XML.
"""

from __future__ import annotations

import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path

# Headless CPU rendering through EGL when the system has it; otherwise the overview renders
# are skipped (the head camera renders on the GPU through mujoco_warp either way).
if "MUJOCO_GL" not in os.environ and ctypes.util.find_library("EGL"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np
import torch
import tyro
from PIL import Image

from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.layout import layout_term


@dataclass
class Args:
    num_envs: int = 8
    steps: int = 50  # control steps (50 Hz) holding the default pose
    randomize: bool = True
    camera: bool = True
    device: str = "cuda:0"
    out_dir: Path = Path("logs/scene_check")
    overview_envs: int = 4
    seed: int = 0


def _overview(env, env_id: int, path: Path, scene_cfg) -> None:
    """Render one world from a fixed third-person camera with the CPU renderer."""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos[env_id].cpu().numpy()
    data.mocap_pos[:] = env.sim.data.mocap_pos[env_id].cpu().numpy()
    data.mocap_quat[:] = env.sim.data.mocap_quat[env_id].cpu().numpy()
    rgba = model.geom_rgba.copy()
    model.geom_rgba[:] = env.sim.model.geom_rgba[env_id].cpu().numpy()
    mujoco.mj_forward(model, data)
    origin = env.scene.env_origins[env_id].cpu().numpy()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = origin + np.array([0.4, 0.3, 0.6])
    cam.distance, cam.azimuth, cam.elevation = 3.2, 215.0, -28.0
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(path)
    model.geom_rgba[:] = rgba


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_args = CanPnpEnvCfgArgs(
        num_envs=args.num_envs, randomize=args.randomize, camera=args.camera, seed=args.seed
    )
    env = make_env(cfg_args, device=args.device)
    env.reset()
    geo = cfg_args.scene.geometry
    term = layout_term(env)
    origins = env.scene.env_origins
    can, shelf_can, robot = env.scene["can"], env.scene["shelf_can"], env.scene["robot"]

    print(f"model: nbody={env.sim.mj_model.nbody} ngeom={env.sim.mj_model.ngeom} "
          f"nq={env.sim.mj_model.nq} nu={env.sim.mj_model.nu} neq={env.sim.mj_model.neq}")
    print(f"actions: {env.action_manager.total_action_dim}  "
          f"obs: { {k: tuple(v.shape) for k, v in env.observation_manager.compute().items()} }")
    env.scene.write(args.out_dir / "scene")

    can0 = can.data.root_link_pos_w.clone() - origins
    shelf0 = shelf_can.data.root_link_pos_w.clone() - origins
    box_floor = geo.box_bottom_z + geo.box_wall_thickness
    failures: list[str] = []
    for i in range(args.num_envs):
        L = {k: v[i].cpu().numpy().round(3).tolist() for k, v in term.layout.items()}
        box_xy = (term.box_center_w[i, :2] - origins[i, :2]).cpu().numpy()
        print(f"[env {i}] table_front_x={L['table_front_x']} table_y={L['table_center_y']} "
              f"robot_xy={L['robot_xy']} yaw={L['robot_yaw_deg']} can={L['can_xyz']} "
              f"shelf_can={shelf0[i].cpu().numpy().round(3).tolist()} "
              f"goal={(term.goal_pos_w[i] - origins[i]).cpu().numpy().round(3).tolist()} "
              f"box_xy={box_xy.round(3).tolist()}")
        if abs(can0[i, 2].item() - geo.can_height / 2 - geo.table_height) > 1e-3:
            failures.append(f"env {i}: can bottom not on tabletop")
        if abs(shelf0[i, 2].item() - geo.can_height / 2 - box_floor) > 2e-3:
            failures.append(f"env {i}: resident can bottom {shelf0[i, 2].item() - geo.can_height / 2:.4f} != box floor {box_floor:.4f}")
        inner = np.array(geo.upper_box_size[:2]) / 2 - geo.box_wall_thickness - geo.can_radius
        if np.any(np.abs(shelf0[i, :2].cpu().numpy() - box_xy) > inner + 1e-6):
            failures.append(f"env {i}: resident can outside box interior")
        if L["robot_xy"][0] > L["table_front_x"] - 0.3:
            failures.append(f"env {i}: robot spawned within 0.3 m of the table")

    if args.camera:
        rgb = env.scene["head_camera"].data.rgb
        for i in range(args.num_envs):
            Image.fromarray(rgb[i].cpu().numpy()).save(args.out_dir / f"head_camera_env{i}.png")
    try:
        for i in range(min(args.overview_envs, args.num_envs)):
            _overview(env, i, args.out_dir / f"overview_env{i}.png", cfg_args.scene)
    except (AttributeError, ImportError, mujoco.FatalError) as error:
        # The CPU renderer needs an EGL/OSMesa library; the head camera renders on the GPU.
        print(f"overview images skipped: no OpenGL context for MUJOCO_GL ({error})")

    zero = torch.zeros(args.num_envs, env.action_manager.total_action_dim, device=env.device)
    max_can_move = torch.zeros(args.num_envs, device=env.device)
    max_shelf_move = torch.zeros(args.num_envs, device=env.device)
    terminated = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(args.steps):
        _, _, term_buf, _, _ = env.step(zero)
        terminated |= term_buf.bool()
        alive = ~terminated
        moved = torch.linalg.norm(can.data.root_link_pos_w - origins - can0, dim=-1)
        moved_shelf = torch.linalg.norm(shelf_can.data.root_link_pos_w - origins - shelf0, dim=-1)
        max_can_move = torch.where(alive, torch.maximum(max_can_move, moved), max_can_move)
        max_shelf_move = torch.where(alive, torch.maximum(max_shelf_move, moved_shelf), max_shelf_move)
    pelvis_z = (robot.data.root_link_pos_w[:, 2] - origins[:, 2]).cpu().numpy().round(3)
    print(f"after {args.steps} steps holding the default pose:")
    print(f"  max can displacement      {max_can_move.cpu().numpy().round(4).tolist()}")
    print(f"  max resident displacement {max_shelf_move.cpu().numpy().round(4).tolist()}")
    print(f"  pelvis z                  {pelvis_z.tolist()}  terminated={terminated.cpu().tolist()}")
    contacts = env.scene["finger_l_can_contacts"].data.found
    print(f"  finger/can contacts found {contacts.reshape(args.num_envs, -1).sum(-1).cpu().tolist()}")
    if bool((max_shelf_move > 5e-3).any()):
        failures.append("resident can moved more than 5 mm in the box")
    if args.camera:
        rgb = env.scene["head_camera"].data.rgb
        Image.fromarray(rgb[0].cpu().numpy()).save(args.out_dir / "head_camera_env0_after.png")

    print("\n".join(["FAIL " + f for f in failures]) if failures else "ALL SCENE CHECKS PASSED")
    print(f"outputs in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
