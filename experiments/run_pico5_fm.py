"""Run a visual five-point FM over the packaged Pico 2.5 actor on flat ground."""
import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
FM = REPO / "downloads/5points_FM"
PICO = REPO.parent / "pico5_student25_sim2sim_20261123"
sys.path[:0] = [str(REPO), str(PICO), str(FM / "inference_source"), str(REPO / "tasks/can_pnp/src")]

from can_pnp.scene.config import HeadCameraCfg
from humanoid_diffusion.keypoints import KP_BODY_NAMES, KP_NAMES, keypoints_to_world
from humanoid_diffusion.model import load_policy
from experiments.pico_markers import PicoMarkers
from pico5.sim2sim import COLORS, ONNX, Playback, StandingFrames


class FMPlayback(Playback):
    def reset(self):
        # The FM supplies one new reference per tick. The bundled player indexes
        # these points to assemble exactly its usual causal 11-frame history.
        self.points = np.full((self.args.steps + 2, 5, 8), np.nan, np.float32)
        pose = dict(rootPosition=[0., 0., .79], rootQuaternionWxyz=[1., 0., 0., 0.],
                    jointPosition=self.defaultJointPosition.copy())
        self.motion = dict(frames=StandingFrames(len(self.points) - 1, pose), initialPose=pose)
        self.loaded_motion_id = self.motion_id
        super().reset()


def fm_state(player, ids, qids):
    d = player.d
    pelvis = Rotation.from_matrix(d.xmat[ids[0]].reshape(3, 3))
    values = [d.qpos[qids]]
    for wrist in ids[3:]:
        relative = pelvis.inv() * Rotation.from_matrix(d.xmat[wrist].reshape(3, 3))
        values += [pelvis.inv().apply(d.xpos[wrist] - d.xpos[ids[0]]),
                   relative.as_quat(canonical=True, scalar_first=True)]
    # This G1 has rigid hands. Represent both grippers as open in the FM state.
    return torch.from_numpy(np.concatenate([*values, pelvis.inv().apply([0., 0., -1.]),
                                            [.065, .065]]).astype(np.float32))


def install_ego_camera(player):
    cfg = HeadCameraCfg()
    m = player.m
    camera = m.camera("tracking").id
    m.cam_bodyid[camera] = m.body("torso_link").id
    m.cam_mode[camera] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    m.cam_pos[camera], m.cam_quat[camera], m.cam_fovy[camera] = cfg.pos, cfg.quat, cfg.fovy
    m.vis.map.znear = .01 / m.stat.extent
    mujoco.mj_forward(m, player.d)
    return camera


def draw_targets(scene, targets):
    for pos, color in zip(targets[PicoMarkers.order], COLORS, strict=True):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .026), pos,
                            np.eye(3).ravel(), np.array(color, np.float32))
        scene.ngeom += 1


@torch.inference_mode()
def rollout(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    fm, config = load_policy(args.fm, device="cpu")
    ds = config["dataset"]
    if (fm.obs_horizon, fm.obs_dim, fm.action_dim) != (2, 48, 47):
        raise ValueError("Expected the G1 visual five-point FM checkpoint")
    steps = round(args.seconds * 50)
    cfg = SimpleNamespace(onnx=ONNX, threads=args.threads, terrain="flat", level=0,
                          mouse_spawn=False, motion=0, start_frame=0, friction=.9,
                          spawn_x=None, heading_deg=0., steps=steps)
    player = FMPlayback(cfg, [dict(id="five-point FM", motion_id=0)])
    m, d = player.m, player.d
    ids = np.array([m.body(name).id for name in KP_BODY_NAMES])
    qids = [m.joint(n).qposadr[0] for n in ds["feature_names"]["observation.state"]]
    markers = PicoMarkers(m, d, ids)
    ego_cam = install_ego_camera(player)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    positions, targets, heights = [], [], []
    replans = 0
    start = d.qpos[:3].copy()
    with ExitStack() as stack:
        ego = stack.enter_context(mujoco.Renderer(m, height=240, width=320))
        overview = stack.enter_context(mujoco.Renderer(m, height=360, width=480))
        view = mujoco.MjvCamera()
        view.azimuth, view.elevation, view.distance = 110, -10, 2.55
        writer = imageio_ffmpeg.write_frames(str(out / "rollout.mp4"), (480, 360), fps=25,
                                             codec="libx264", macro_block_size=1,
                                             output_params=["-movflags", "+faststart"])
        writer.send(None)
        stack.callback(writer.close)
        for i in range(steps):
            offset = i % fm.execute_steps
            # Only the last two RGB/state frames at a replan enter the FM.
            if offset == 0 or offset == fm.execute_steps - 1:
                ego.update_scene(d, camera=ego_cam)
                rgb = torch.from_numpy(ego.render()).permute(2, 0, 1)[None].float()
                rgb = torch.nn.functional.interpolate(rgb, (ds["image_size"],) * 2, mode="area").round().byte()[0]
                state = fm_state(player, ids, qids)
                if offset == 0:
                    images = torch.stack([previous_rgb if i else rgb, rgb])[None]
                    states = torch.stack([previous_state if i else state, state])[None]
                    pred = fm.predict_action(images, states)
                    anchor = torch.tensor(d.qpos[:3], dtype=torch.float32)[None]
                    quat = torch.tensor(d.qpos[3:7], dtype=torch.float32)[None]
                    pos, rot = keypoints_to_world(pred["body_action_pred"], anchor, quat)
                    plan_pos, plan_rot = pos[0].numpy(), rot[0].numpy()
                    replans += 1
                previous_rgb, previous_state = rgb, state
            goal = plan_pos[offset]
            player.points[i] = markers.encode(goal, plan_rot[offset])
            player.step()
            positions.append(d.xpos[ids].copy())
            targets.append(goal.copy())
            heights.append(float(d.qpos[2] - player.spawn_height))
            if i % 2 == 0:
                view.lookat[:] = [*d.qpos[:2], .78 + player.spawn_height]
                overview.update_scene(d, camera=view)
                for light in overview.scene.lights[:overview.scene.nlight]:
                    if light.headlight:
                        light.ambient[:] = .48
                        light.diffuse[:] = .8
                draw_targets(overview.scene, goal)
                writer.send(overview.render())
            if heights[-1] < .35:
                break
    errors = np.linalg.norm(np.asarray(targets) - positions, axis=-1)
    np.savez_compressed(out / "trace.npz", target_world_m=targets, measured_world_m=positions,
                        time_post_step_s=np.arange(1, len(errors) + 1) / 50, links=KP_BODY_NAMES)
    result = dict(fm=str(args.fm), pico=str(ONNX), terrain="flat", seed=args.seed,
                  requested_seconds=args.seconds, simulated_seconds=len(errors) / 50,
                  replans=replans, fell_below_035m=heights[-1] < .35, min_pelvis_height_m=min(heights),
                  displacement_xy_m=(d.qpos[:2] - start[:2]).tolist(),
                  mean_position_error_m=float(errors.mean()), p95_position_error_m=float(np.percentile(errors, 95)),
                  by_point_mean_m=dict(zip(KP_NAMES, errors.mean(axis=0).tolist())))
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fm", type=Path, default=FM / "fivepoints_fm.pt")
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or round(args.seconds * 50) < 1 or args.threads < 1:
        parser.error("Specify a positive duration and thread count.")
    rollout(args)
