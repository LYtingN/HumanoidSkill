"""Dense reset starts along every collected episode, from the LeRobot export and a physical replay.

    PYTHONPATH=hskill/src:tasks/can_pnp/src python tasks/can_pnp/scripts/extract_dense_starts.py \\
        data/residual_rl/can_pnp tasks/can_pnp/src/can_pnp/scene/assets/grasp_starts_dense.json \\
        [--stride-s 0.5]

``grasp_starts.json`` holds one recorded state per phase start. The raw MuJoCo collection with the
can poses of every frame is not available, but the LeRobot export records the robot of every frame
exactly (root pose from ``derived/keypoints_v1``, joints from ``observation.state``: identical to
the phase starts) and the gripper commands. The can is still on the table until the hand reaches it,
so frames before CLOSE keep the recorded resting pose. From the CLOSE start on, the episode is
replayed: the robot body follows the recording kinematically, the grippers track the recorded
commands with their actuators and the can is a free body, so its pose in the hand is physical. At
every recorded phase start the can is put back to its recorded pose, so drift stays within one phase.
Every ``stride`` frames up to the episode's RELEASE start becomes a start if the replayed state is
physically valid: before LIFT the can stands near its recorded table pose; from LIFT on both fingers
touch it and it is within reach of the gripper (a replay that let the can slip is dropped there). The
recorded phase starts are kept exactly. The replay is checked against the recorded can at each phase
start (the pose the replay reached before it was put back).

Output: ``{joint_names, stride_frames, episodes: {episode: {table_front_x, table_center_y,
replay_check_m, starts: [{frame, phase, phase_start, root_pos, root_quat, joint_pos, can_pos, can_quat,
shelf_can_pos, shelf_can_quat, gripper_opening}]}}}``: ``joint_pos`` in ``joint_names`` order,
``gripper_opening`` measured (left, right) in metres, poses in the env frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PHASES = ("APPROACH", "REACH", "CLOSE", "LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE")


def load_episode(root: Path, episode: int, names: list[str], body_joints: tuple[str, ...]):
    df = pd.read_parquet(root / f"data/chunk-000/episode_{episode:06d}.parquet")
    kp = np.load(root / f"derived/keypoints_v1/episode_{episode:06d}.npz")
    state = np.stack(df["observation.state"].to_numpy())
    col = [names.index(n) for n in body_joints]
    grip = lambda key: np.stack([np.atleast_1d(v)[0] for v in df[key].to_numpy()]).astype(np.float64)
    return dict(
        root_pos=kp["pos_w"][:, 0].astype(np.float64), root_quat=kp["quat_w"][:, 0].astype(np.float64),
        joints=state[:, col].astype(np.float64),
        measured=np.stack((grip("observation.left_gripper"), grip("observation.right_gripper")), -1),
        command=np.stack((grip("action.left_gripper"), grip("action.right_gripper")), -1))


def quat_angvel(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity (rad/s) taking unit wxyz quaternions q0 to q1 in dt."""
    w0, v0 = q0[..., :1], q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    # d = q1 * conj(q0)
    dw = w1 * w0 + (v1 * v0).sum(-1, keepdims=True)
    dv = -w1 * v0 + w0 * v1 - np.cross(v1, v0)
    sign = np.where(dw < 0, -1.0, 1.0)
    dw, dv = dw * sign, dv * sign
    angle = 2 * np.arctan2(np.linalg.norm(dv, axis=-1, keepdims=True), dw)
    axis = dv / np.maximum(np.linalg.norm(dv, axis=-1, keepdims=True), 1e-12)
    return axis * angle / dt


def nlerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    q = (1 - a) * q0 + a * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lerobot", type=Path, help="LeRobot can_pnp root (data/, derived/keypoints_v1/, meta/)")
    p.add_argument("out", type=Path)
    p.add_argument("--grasp-starts", type=Path, default=None, help="Recorded phase starts (default: the asset)")
    p.add_argument("--stride-s", type=float, default=0.5, help="Seconds between dense starts")
    p.add_argument("--episodes", type=int, nargs="*", help="Only these episodes (default: all)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from can_pnp.scene.config import CanPnpSceneCfg, SceneGeometryCfg
    from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
    from can_pnp.starts import GRASP_STARTS, GraspStarts, apply_grasp_start
    from hskill.embodiment import g1_aloha as C
    from hskill.s0.controller import gripper_opening, gripper_q

    recorded = json.loads((args.grasp_starts or GRASP_STARTS).read_text())
    names = json.loads((args.lerobot / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    episodes = sorted(int(e) for e in recorded) if not args.episodes else sorted(args.episodes)
    data = {e: load_episode(args.lerobot, e, names, C.BODY_JOINT_NAMES) for e in episodes}
    frame_of = {e: {ph: int(recorded[str(e)]["phases"][ph]["frame"]) for ph in PHASES} for e in episodes}
    for e in episodes:  # the export reproduces the recorded robot at every phase start
        for ph in PHASES:
            r, f = recorded[str(e)]["phases"][ph], frame_of[e][ph]
            err = max(np.abs(data[e]["root_pos"][f] - r["root_pos"]).max(),
                      np.abs(data[e]["joints"][f] - [r["joint_pos"][n] for n in C.BODY_JOINT_NAMES]).max())
            if err > 1e-5:
                raise ValueError(f"episode {e} {ph}: LeRobot robot state differs from the recorded start by {err}")

    n = len(episodes)
    geometry = SceneGeometryCfg(can_mass=.1)
    env = make_env(CanPnpEnvCfgArgs(num_envs=n, randomize=False, camera=False, seed=0, enable_time_out=False,
                                    scene=CanPnpSceneCfg(geometry=geometry)), device=args.device)
    device = env.device
    robot, can = env.scene["robot"], env.scene["can"]
    fingers = [env.scene[f"finger_{side}_can_contacts"] for side in ("l", "r")]
    eef = robot.body_names.index(C.KEYPOINT_BODIES[C.RIGHT_EEF])
    body_ids = robot.find_joints(C.BODY_JOINT_NAMES, preserve_order=True)[0]
    gripper_ids = robot.find_joints(C.GRIPPER_JOINT_NAMES, preserve_order=True)[0]
    ids = torch.arange(n, device=device)
    origin = env.scene.env_origins
    env.reset()
    close = GraspStarts(args.grasp_starts or GRASP_STARTS, phase="CLOSE")
    apply_grasp_start(env, ids, close, torch.tensor([close.episode_ids.index(e) for e in episodes]))
    dt, substeps = env.step_dt, env.cfg.decimation
    start = np.array([frame_of[e]["CLOSE"] for e in episodes])
    stop = np.array([frame_of[e]["RELEASE"] for e in episodes])
    stride = max(1, round(args.stride_s / dt))
    replay = {e: {} for e in episodes}  # frame -> (can pos, can quat, opening, both fingers on it, gripper-can m)
    anchors = {i: {frame_of[e][ph]: recorded[str(e)]["phases"][ph] for ph in ("LIFT", "TURN_WALK", "PLACE_REACH", "LOWER")}
               for i, e in enumerate(episodes)}
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

    def robot_at(k: int, a: float):
        """Root state and body joints of every world between its frames f and f + 1 (a in [0, 1])."""
        f = np.minimum(start + k, stop)
        g = np.minimum(f + 1, stop)
        get = lambda key, frames: np.stack([data[e][key][fr] for e, fr in zip(episodes, frames)])
        p0, p1 = get("root_pos", f), get("root_pos", g)
        q0, q1 = get("root_quat", f), get("root_quat", g)
        j0, j1 = get("joints", f), get("joints", g)
        moving = (f < stop)[:, None]
        lin = np.where(moving, (p1 - p0) / dt, 0.)
        ang = np.where(moving, quat_angvel(q0, q1, dt), 0.)
        root = np.concatenate(((1 - a) * p0 + a * p1, nlerp(q0, q1, a), lin, ang), -1)
        return root, (1 - a) * j0 + a * j1, np.where(moving, (j1 - j0) / dt, 0.), get("command", f)

    def record(k: int):
        env.sim.forward()
        env.scene.update(dt)
        f = np.minimum(start + k, stop)
        pos = (can.data.root_link_pos_w - origin).cpu().numpy()
        quat = can.data.root_link_quat_w.cpu().numpy()
        opening = gripper_opening(robot.data.joint_pos[:, gripper_ids]).cpu().numpy()
        both = torch.stack([fg.data.found.flatten(1).gt(0).any(-1) for fg in fingers]).all(0).cpu().numpy()
        reach = (robot.data.body_link_pos_w[:, eef] - can.data.root_link_pos_w).norm(dim=-1).cpu().numpy()
        put_back = []
        for i, e in enumerate(episodes):
            if f[i] <= stop[i] and f[i] not in replay[e]:
                replay[e][int(f[i])] = (pos[i], quat[i], opening[i], bool(both[i]), float(reach[i]))
                if int(f[i]) in anchors[i]:
                    put_back.append(i)
        if put_back:
            r = [anchors[i][int(f[i])] for i in put_back]
            state = torch.zeros(len(put_back), 13, device=device)
            state[:, :3] = tensor([x["can_pos"] for x in r]) + origin[put_back]
            state[:, 3:7] = tensor([x["can_quat"] for x in r])
            can.write_root_state_to_sim(state, env_ids=torch.tensor(put_back, device=device))

    record(0)
    for k in range(int((stop - start).max())):
        for s in range(substeps):
            root, joints, joint_vel, command = robot_at(k, (s + 1) / substeps)
            root_t = tensor(root)
            root_t[:, :3] += origin
            robot.write_root_state_to_sim(root_t, env_ids=ids)
            robot.write_joint_state_to_sim(tensor(joints), tensor(joint_vel), joint_ids=body_ids, env_ids=ids)
            target = robot.data.joint_pos_target.clone()
            target[:, body_ids] = tensor(joints)
            target[:, gripper_ids] = gripper_q(tensor(command))
            robot.set_joint_position_target(target)
            env.scene.write_data_to_sim()
            env.sim.step()
        record(k + 1)

    # Check the replayed can against the recorded later phase starts.
    report = {}
    for ph in ("LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE"):
        dp, da = [], []
        for e in episodes:
            r = recorded[str(e)]["phases"][ph]
            pos, quat = replay[e][frame_of[e][ph]][:2]
            dp.append(np.linalg.norm(pos - np.array(r["can_pos"])))
            dot = abs(float(np.dot(quat, r["can_quat"])))
            da.append(np.degrees(2 * np.arccos(min(1.0, dot))))
        report[ph] = dict(pos_cm_p50=100 * float(np.median(dp)), pos_cm_p90=100 * float(np.percentile(dp, 90)),
                          pos_cm_max=100 * float(np.max(dp)), angle_deg_p50=float(np.median(da)),
                          angle_deg_p90=float(np.percentile(da, 90)))
    print("Replayed can vs recorded phase starts:", json.dumps(report, indent=1), flush=True)

    def valid(e: int, f: int) -> bool:
        pos, quat, _, both, reach = replay[e][f]
        axis_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
        if f < frame_of[e]["LIFT"]:  # being grasped on the table
            rest = np.array(recorded[str(e)]["phases"]["CLOSE"]["can_pos"])
            return bool(np.linalg.norm(pos - rest) < 0.03 and axis_z > 0.9 and reach < 0.15)
        return bool(both and reach < 0.11)

    out, kept = {}, {ph: [0, 0] for ph in PHASES}
    for i, e in enumerate(episodes):
        rec = recorded[str(e)]
        frames = sorted(set(range(0, frame_of[e]["RELEASE"] + 1, stride)) | set(frame_of[e].values()))
        starts = []
        for f in frames:
            phase = max((ph for ph in PHASES if frame_of[e][ph] <= f), key=lambda ph: frame_of[e][ph])
            exact = f == frame_of[e][phase]
            r = rec["phases"][phase]
            if not exact and f >= frame_of[e]["CLOSE"]:
                kept[phase][1] += 1
                if not valid(e, f):
                    continue
                kept[phase][0] += 1
            if f < frame_of[e]["CLOSE"]:
                can_pos, can_quat = r["can_pos"], r["can_quat"]  # resting on the table
            elif exact and phase != "CLOSE":
                can_pos, can_quat = r["can_pos"], r["can_quat"]
            else:
                pos, quat = replay[e][f][:2]
                can_pos, can_quat = pos.tolist(), quat.tolist()
            starts.append(dict(
                frame=int(f), phase=phase, phase_start=bool(exact),
                root_pos=data[e]["root_pos"][f].tolist(), root_quat=data[e]["root_quat"][f].tolist(),
                joint_pos=data[e]["joints"][f].tolist(),
                can_pos=list(map(float, can_pos)), can_quat=list(map(float, can_quat)),
                shelf_can_pos=r["shelf_can_pos"], shelf_can_quat=r["shelf_can_quat"],
                gripper_opening=data[e]["measured"][f].tolist()))
        out[str(e)] = dict(table_front_x=rec["table_front_x"], table_center_y=rec["table_center_y"],
                           replay_check_m={ph: float(np.linalg.norm(replay[e][frame_of[e][ph]][0] - np.array(rec["phases"][ph]["can_pos"])))
                                         for ph in ("LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE")},
                           starts=starts)
    def rounded(x):
        if isinstance(x, float):
            return round(x, 6)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rounded(v) for v in x]
        return x

    doc = dict(joint_names=list(C.BODY_JOINT_NAMES), stride_frames=stride, episodes=out)
    args.out.write_text(json.dumps(rounded(doc), separators=(",", ":")))
    total = sum(len(v["starts"]) for v in out.values())
    print("Replayed frames kept as starts per phase (kept / replayed): "
          + ", ".join(f"{ph} {k}/{n}" for ph, (k, n) in kept.items() if n), flush=True)
    print(f"Wrote {total} starts for {len(out)} episodes (stride {stride} frames) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
