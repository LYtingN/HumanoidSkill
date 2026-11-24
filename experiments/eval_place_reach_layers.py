"""Measure can execution and held-can FM intent from recorded PLACE_REACH starts."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for source in ("supermimic-reproduced/src", "downloads/5points_FM/inference_source",
               "hskill/src", "tasks/can_pnp/src", "downloads/morphmimic-v0.1/source"):
    sys.path.insert(0, str(ROOT / source))
os.environ.setdefault("WARP_CACHE_PATH", str(ROOT / ".cache/warp-cu130"))

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from can_pnp.scene.config import CanPnpSceneCfg, HeadCameraCfg, SceneGeometryCfg
from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.scene.layout import layout_term
from can_pnp.starts import GraspStarts, apply_grasp_start
from can_pnp.task_spec import TaskCfg, TaskTracker
from hskill.learners.flashsac.network import FlashSACActor
from hskill.s0.official_five_batch import BatchedOfficialFive
from humanoid_diffusion.keypoints import KP_BODY_NAMES


@torch.inference_mode()
def evaluate(args):
    torch.manual_seed(args.seed)
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if weights["algorithm"] != "flashsac":
        raise ValueError("Expected a five-point FlashSAC noise actor")
    cfg = weights["config"]
    actor = FlashSACActor(cfg["blocks"], cfg["obs_dim"], cfg["actor_width"], cfg["action_dim"], cfg.get("prior_std"))
    actor.load_state_dict(weights["actor"])
    actor = actor.to("cuda:0").eval()
    starts = GraspStarts(phase="PLACE_REACH", measured_gripper=weights["measured_gripper"])
    rows = np.random.default_rng(args.seed).choice(len(starts), args.episodes, replace=False)
    geo = SceneGeometryCfg(can_mass=.1)
    env = make_env(CanPnpEnvCfgArgs(num_envs=args.batch, randomize=False, seed=args.seed,
                                    enable_time_out=False, nan_termination=True,
                                    scene=CanPnpSceneCfg(geometry=geo, camera=HeadCameraCfg(width=320, height=240))),
                   device="cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        ctrl = BatchedOfficialFive(env, ROOT / "downloads/5points_FM/fivepoints_fm.pt",
                                   ROOT / "downloads/morphmimic-v0.1", fm_batch=min(8, args.batch),
                                   gripper_hysteresis=weights["gripper_hysteresis"],
                                   tracker_window=weights.get("tracker_window", "spread"))
        fm = ctrl.fm.eval()
        frames = 1 if weights["noise_repeat"] else fm.horizon
        assert cfg["action_dim"] == frames * fm.action_dim
        task_cfg = TaskCfg(upright_release=weights["upright_release"], throw_speed=weights["throw_speed"])
        wrist = ctrl.robot.body_names.index(KP_BODY_NAMES[4])
        robot = ctrl.robot.data
        can = env.scene["can"].data
        fingers = [env.scene[f"finger_{side}_can_contacts"] for side in ("l", "r")]
        ids = torch.arange(args.batch, device=env.device)
        ticks = round(8 / env.step_dt)
        stride = round(.1 / env.step_dt)
        for start in range(0, len(rows), args.batch):
            file = args.output / f"batch_{start:03d}.npz"
            batch = rows[start:start + args.batch]
            n = len(batch)
            batch = np.pad(batch, (0, args.batch - n), mode="edge")
            episodes = np.asarray(starts.episode_ids)[batch[:n]]
            ctrl.render = True
            env.reset()
            apply_grasp_start(env, ids, starts, torch.from_numpy(batch))
            ctrl.reset_worlds(ids)
            layout = layout_term(env)
            rim = layout.box_center_w.clone()
            rim[:, 2] = env.scene.env_origins[:, 2] + geo.box_bottom_z + geo.upper_box_size[2]
            task = TaskTracker(task_cfg, geo, layout.box_center_w.clone(), layout.goal_pos_w.clone(),
                               env.scene.env_origins[:, 2], can.root_link_pos_w, env.step_dt)
            live = ids < n
            outcome = np.full(args.batch, "timeout", dtype="U24")
            end_time = np.full(args.batch, ticks * env.step_dt)
            actual = np.full((ticks // stride + 1, args.batch, 3), np.nan, np.float32)
            intent = np.full_like(actual, np.nan)
            actual[0] = (can.root_link_pos_w - rim).cpu().numpy()
            def contact():
                return torch.stack([f.data.found.flatten(1).gt(0).any(1) for f in fingers]).all(0)

            initial_hold = (live & contact()).cpu().numpy()
            intent[0, initial_hold] = actual[0, initial_hold]
            # Follow the same physical object after scoring. An mjlab auto-reset
            # starts a different world, so permanently censor that trajectory.
            physical = live.clone()
            follow = np.full_like(actual, np.nan)
            follow[0] = actual[0]
            wrist_poses = np.full((*actual.shape[:2], 2, 7), np.nan, np.float32)
            contact_samples = np.zeros(actual.shape[:2], bool)
            contact_samples[0] = initial_hold
            observed0 = torch.cat((robot.body_link_pos_w[:, wrist] - rim,
                                   robot.body_link_quat_w[:, wrist]), dim=-1).cpu().numpy()
            wrist_poses[0, :, 0] = wrist_poses[0, :, 1] = observed0
            for i in range(ticks):
                k = i % fm.execute_steps
                if k == 0:
                    cond = ctrl.encode()
                    mean, _ = actor.get_mean_and_std(cond.flatten(1), training=False)
                    latent = weights["noise_scale"] * mean.tanh().reshape(args.batch, frames, fm.action_dim)
                    ctrl.set_plan(ctrl.sample(cond, latent.expand(-1, fm.horizon, -1)))
                ctrl.render = k >= fm.execute_steps - fm.obs_horizon
                cmd = ctrl.execute_plan()
                _, _, term, trunc, _ = env.step(cmd)
                physical &= ~(term | trunc | ctrl.nonfinite)
                auto = live & (term | trunc | ctrl.nonfinite)
                valid = live & ~auto
                held = contact()
                task.step(can.root_link_pos_w, can.root_link_quat_w,
                          ctrl.commanded_width[:, 1] > task_cfg.release_opening, valid, held)
                if (i + 1) % stride == 0:
                    sample = (i + 1) // stride
                    present = physical.cpu().numpy()
                    follow[sample, present] = (can.root_link_pos_w - rim)[physical].cpu().numpy()
                    contact_samples[sample] = (physical & held).cpu().numpy()
                    measured_pose = torch.cat((robot.body_link_pos_w[:, wrist] - rim,
                                               robot.body_link_quat_w[:, wrist]), dim=-1)
                    reference_pose = torch.cat((ctrl.plan_pos[:, k, 4] - rim, ctrl.plan_quat[:, k, 4]), dim=-1)
                    wrist_poses[sample, present] = torch.stack((measured_pose, reference_pose), 1)[physical].cpu().numpy()
                    actual[sample, valid.cpu().numpy()] = (can.root_link_pos_w - rim)[valid].cpu().numpy()
                    grip = quat_apply_inverse(robot.body_link_quat_w[:, wrist],
                                              can.root_link_pos_w - robot.body_link_pos_w[:, wrist])
                    ref = ctrl.plan_pos[:, k, 4] + quat_apply(ctrl.plan_quat[:, k, 4], grip) - rim
                    keep = valid & held
                    intent[sample, keep.cpu().numpy()] = ref[keep].cpu().numpy()
                for label, event in (("env_termination", auto), ("placed", task.placed),
                                     ("toppled", task.toppled), ("thrown", task.thrown)):
                    ended = live & event
                    mask = ended.cpu().numpy()
                    outcome[mask], end_time[mask] = label, (i + 1) * env.step_dt
                    live = live & ~ended
                if not physical.any():
                    break
                if ctrl.render:
                    ctrl.push_history()
            np.savez_compressed(file, episodes=episodes, executed=actual[:, :n], intent=intent[:, :n],
                                time=np.arange(len(actual)) * stride * env.step_dt, outcome=outcome[:n],
                                entered=task.entered[:n].cpu().numpy(), end_time=end_time[:n],
                                follow=follow[:, :n], contact=contact_samples[:, :n], wrist_poses=wrist_poses[:, :n],
                                tracker_window=ctrl.tracker_window, tracker_indices=ctrl.tracker.reference.window_indices(0),
                                noise_repeat=weights["noise_repeat"], noise_scale=weights["noise_scale"],
                                action_dim=cfg["action_dim"], checkpoint=str(args.checkpoint),
                                step=weights["step"], seed=args.seed)
            print(f"{start + n}/{len(rows)} episodes: {np.unique(outcome[:n], return_counts=True)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "logs/remote_ckpt/repeat_step4000.pt")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    evaluate(p.parse_args())
