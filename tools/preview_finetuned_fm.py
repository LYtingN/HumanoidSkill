"""Render the DSRL-steered FM on recorded can-pick-and-place observations."""
import argparse
import json
import os
import sys
from html import escape
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "hskill/src"), str(ROOT / "tasks/can_pnp/src"), str(ROOT / "downloads/ODE/inference_source")]

import cv2
import imageio_ffmpeg
import mujoco
import numpy as np
import pyarrow.parquet as pq
import torch

from can_pnp.scene.constants import ASSETS_DIR
from hskill.embodiment import g1_aloha as C
from hskill.s1.noise_policy import NoisePolicy, SteeredDiffusionSource
from humanoid_diffusion.data import open_video, quat_to_matrix, read_matrix
from humanoid_diffusion.model import load_policy


def absolute_quat(anchor_rotation, rotation6d):
    first, second = rotation6d.reshape(3, 2).T
    x = first / np.linalg.norm(first)
    y = second - x.dot(second) * x
    y /= np.linalg.norm(y)
    rotation = anchor_rotation @ np.column_stack([x, y, np.cross(x, y)])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.flatten().astype(np.float64))
    return quat


def label(image, message, origin, size=0.6):
    cv2.putText(image, message, origin, cv2.FONT_HERSHEY_SIMPLEX, size,
                (230, 236, 245), 1, cv2.LINE_AA)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--phase", default="CLOSE")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/fm-preview/best_cycle1885.mp4")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    weights = ROOT / "downloads"
    policy, config = load_policy(weights / "supermimic-fm/can_pnp_supermimic_flow_matching_best_ema.pt", device="cpu")
    checkpoint = torch.load(weights / "DSRL_finetune_20261116/best_cycle1885.pt", map_location="cpu", weights_only=False)
    noise = NoisePolicy(policy.n_tokens, policy.n_emb, policy.horizon, policy.action_dim,
                        hidden=checkpoint["args"]["hidden"])
    noise.load_state_dict(checkpoint["noise_policy"])
    source = SteeredDiffusionSource(policy, noise.eval(), mode="sample", record=False)
    steps = checkpoint["args"]["execute_steps"]
    torch.manual_seed(args.seed)
    cfg = config["dataset"]
    dataset = ROOT / "data/residual_rl/can_pnp"
    info = json.loads((dataset / "meta/info.json").read_text())
    fmt = dict(episode_index=args.episode, episode_chunk=args.episode // info.get("chunks_size", 1000),
               video_key=cfg["image_key"])
    table = pq.read_table(dataset / info["data_path"].format(**fmt))
    selections = {key: [info["features"][key]["names"].index(name) for name in names]
                  for key, names in cfg["feature_names"].items()}
    states = torch.from_numpy(read_matrix(table, cfg["observation_keys"], selections))
    joint_names = cfg["feature_names"]["observation.state"]
    assert list(joint_names) == list(C.BODY_JOINT_NAMES) and cfg["action_keys"][0] == "observation.state"
    roots = quat_to_matrix(read_matrix(table, [cfg["reference_rotation_key"]]))
    start = json.loads((ASSETS_DIR / "grasp_starts.json").read_text())[str(args.episode)]["phases"][args.phase]["frame"]
    stride = cfg["frame_stride"]
    fps = info["fps"]
    end = min(len(table), start + round(fps * args.seconds))
    if end <= start or not 1 <= steps <= policy.horizon:
        raise ValueError("No preview frames or invalid replan interval")
    first = max(0, start - (policy.obs_horizon - 1) * stride)
    cap = open_video(dataset / info["video_path"].format(**fmt))
    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 0.1:
        raise ValueError("Video is unreadable or misaligned with dataset FPS")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    # Only retain the input histories and the frames used for playback.
    frames = {}
    try:
        for i in range(first, end):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Failed to decode video frame {i}")
            if (i - start) % stride == 0 or i == 0:
                frames[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    model = mujoco.MjModel.from_xml_path(str(C.ROBOT_XML))
    data = mujoco.MjData(model)
    joint_ids = [model.joint(name).qposadr.item() for name in joint_names]
    fingers = [[model.joint(name).qposadr.item() for name in names]
               for names in zip(C.GRIPPER_JOINT_NAMES, C.MIRROR_GRIPPER_JOINT_NAMES)]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.82]
    camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -10
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".part.mp4")
    writer = imageio_ffmpeg.write_frames(str(partial), (960, 448), fps=fps / stride, codec="libx264",
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=360, width=480) as renderer:
            for anchor in range(start, end, steps * stride):
                history = np.maximum(0, anchor + np.arange(1 - policy.obs_horizon, 1) * stride)
                size = cfg["image_size"]
                images = np.stack([cv2.resize(frames[i], (size, size), interpolation=cv2.INTER_AREA) for i in history])
                joints, widths, rotations = source.predict(torch.tensor([0]), torch.from_numpy(images).permute(0, 3, 1, 2)[None],
                                                         states[history][None])
                if not all(torch.isfinite(value).all() for value in (joints, widths, rotations)):
                    raise ValueError(f"Nonfinite prediction at frame {anchor}")
                if anchor == start:
                    print(f"Loaded cycle {checkpoint['cycle']} and predicted {joints.shape[1]} poses from frame {start}", flush=True)
                for step, frame in enumerate(range(anchor, min(end, anchor + steps * stride), stride)):
                    data.qpos[joint_ids] = joints[0, step].numpy()
                    data.qpos[3:7] = absolute_quat(roots[anchor], rotations[0, step].numpy())
                    opening = widths[0, step].numpy()
                    for finger_ids, width in zip(fingers, opening):
                        data.qpos[finger_ids] = C.GRIPPER_CLOSED_Q * (1 - np.clip(width / C.GRIPPER_OPEN_WIDTH, 0, 1))
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    canvas = np.full((448, 960, 3), (24, 30, 38), dtype=np.uint8)
                    rgb = frames[frame]
                    scale = min(480 / rgb.shape[1], 360 / rgb.shape[0])
                    eye = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
                    x, y = (480 - eye.shape[1]) // 2, 40 + (360 - eye.shape[0]) // 2
                    canvas[y:y + eye.shape[0], x:x + eye.shape[1]] = eye
                    canvas[40:400, 480:] = renderer.render()
                    label(canvas, "Recorded ego camera", (16, 27))
                    label(canvas, "FM + DSRL   /   predicted G1 pose", (496, 27))
                    label(canvas, f"Episode {args.episode}   |   {args.phase} + {(frame-start)/fps:.2f}s   |   frame {frame}", (16, 430), 0.52)
                    label(canvas, f"Gripper width (mm)     L {opening[0]*1000:.0f}    R {opening[1]*1000:.0f}", (496, 430), 0.52)
                    writer.send(canvas)
    finally:
        writer.close()
    partial.replace(args.output)
    title = f"G1 动作预览 · DSRL cycle {checkpoint['cycle']}"
    args.output.with_suffix(".html").write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title>'
        '<style>body{background:#181e26;color:white;font:16px sans-serif;margin:3vh auto;width:min(96vw,1200px)}'
        'video{display:block;width:100%;max-height:80vh}p{line-height:1.6}</style>'
        f'<video controls autoplay muted loop playsinline src="{escape(args.output.name, quote=True)}"></video>'
        f'<p>第 {args.episode} 条演示 · {escape(args.phase)}。左侧为录制的第一视角，右侧为微调策略预测的 G1 姿态。'
        '机器人骨盆位置固定，夹爪宽度以毫米显示。</p></html>',
        encoding="utf-8",
    )
    print(f"{args.output}  ({len(range(start, end, stride))} frames at {fps/stride:g} fps)", flush=True)


if __name__ == "__main__":
    main()
