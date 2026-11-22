"""Five-point FM plans tracked by the official v0.1 human/five ONNX policy in N mjlab worlds.

All worlds replan together every ``execute_steps`` control steps. The tracker keeps per-world
proprioception history and reference frames in batched NumPy (the release's PolicyRunner
computations over a leading world axis) and runs one batched ONNX inference per control step. With one world it reproduces
:class:`hskill.s0.official_five.OfficialFiveController` driven by explicit FM noise.
"""
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from deploy.sim2sim.human import HUMAN_SPINE3, _UNDO, _anchor, split_frames
from deploy.sim2sim.obs import (GOAL_K, PROPRIO_DIM, axis_angle_quat, quat_apply_inverse as np_rotate_back, quat_inv,
                                quat_mul, rot6d, yaw_quat)
from deploy.sim2sim.policy_node import PolicyRunner, pruned
from humanoid_diffusion.keypoints import KP_NAMES as FM_POINTS, keypoints_to_world
from humanoid_diffusion.model import load_policy

from hskill.embodiment import g1_aloha as C
from hskill.s0.controller import DiffusionObservations, gripper_opening, gripper_q
from hskill.s0.official_five import HumanFiveReference
from hskill.s0.onnx_torch import OnnxTorch

PREVIEW_FRAMES = 10


def human_features_batch(windows, mask):
    """`deploy.sim2sim.human.human_features` of windows [N, K, 168] -> [N, K * 24 * 9]."""
    pos, quat = split_frames(windows)
    a = _anchor(mask)
    yaw = yaw_quat(quat[:, 0, a])[:, None, None]
    origin = np.stack([pos[:, 0, a, 0], pos[:, 0, a, 1], np.zeros(len(pos))], -1)[:, None, None]
    pos = np_rotate_back(yaw, pos - origin)
    return np.concatenate([pos, rot6d(quat_mul(quat_inv(yaw), quat))], -1).reshape(len(windows), -1)


def heading_features_batch(robot_quat, windows, mask):
    """`deploy.sim2sim.human.heading_features` of torso orientations [N, 4] and windows [N, K, 168]."""
    _, quat = split_frames(windows)
    slot = HUMAN_SPINE3 if mask[HUMAN_SPINE3] else _anchor(mask)
    rel = quat_mul(quat_inv(robot_quat)[:, None], quat_mul(quat[:, :, slot], _UNDO))
    return rot6d(rel).reshape(len(windows), -1)


def anchor_quat_batch(robot, imu_quat, q):
    """`Robot.anchor_quat` of IMU orientations [N, 4] and native joint positions [N, J]."""
    n = len(q)
    rel = np.zeros((n, len(robot.fk_parent), 4))
    for b, parent in enumerate(robot.fk_parent):
        rot = (np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), (n, 4)) if parent < 0
               else quat_mul(rel[:, parent], robot.fk_quat[b]))
        for joint, axis, ref in robot.fk_joints[b]:
            rot = quat_mul(rot, axis_angle_quat(np.asarray(axis), q[:, joint] - ref))
        rel[:, b] = rot
    root = quat_mul(imu_quat, quat_inv(rel[:, robot.fk_imu]))
    return quat_mul(root, rel[:, robot.fk_anchor])


class BatchedHumanFive:
    """The v0.1 PolicyRunner's human/five inputs for N robots at once, one ONNX call per step.

    Reproduces per-world ``PolicyRunner(..., interface="human", human_points="five")`` with a
    :class:`HumanFiveReference` that holds ten sampled future frames of each plan.
    """

    def __init__(self, release: Path, n: int, execute_steps: int, horizon: int, device="cpu"):
        export = release / "morphmimic_policy"
        tracked = json.loads((export / "g1_amber.json").read_text())["robot"]["tracked_bodies"]
        stand = release / "source/deploy/vr_teleop/gmr_configs/stand.json"
        # One reference converts every world's frames: its update is independent per frame.
        self.reference = HumanFiveReference(stand, tracked, execute_steps, future_horizon=horizon)
        self.runner = PolicyRunner(export, "g1_amber", self.reference, interface="human", human_points="five")
        self.robot, self.n = self.runner.robot, n
        self.mask = self.runner.point_mask
        self.length = self.runner.history.length
        self.history = np.zeros((n, self.length, PROPRIO_DIM), np.float32)
        self.fresh = np.ones(n, bool)
        self.last_action = np.zeros((n, 29))
        self.windows = self.features = None
        self.default_canonical = self.robot.to_canonical(self.robot.default_joint_pos)
        # On a GPU the ONNX graph runs in PyTorch: onnxruntime on the CPU dominates otherwise.
        self.graph = None
        if torch.device(device).type != "cpu":
            import onnx

            model = pruned(onnx.load(export / "policy.onnx"), "action_human")
            model = onnx.load_from_string(model) if isinstance(model, (bytes, bytearray)) else model
            self.graph = OnnxTorch(model, device)
            self.constants = {k: torch.as_tensor(v, device=device) for k, v in self.runner.inputs.items()}
            self.device_features = None

    def reset(self, ids):
        self.fresh[ids] = True
        self.last_action[ids] = 0.0

    def set_plan(self, pos, quat):
        """World keypoint plans [N, H, 5, 3] and [N, H, 5, 4] (NumPy)."""
        idx = self.reference.future_indices
        n, k = len(pos), len(idx)
        self.reference.update(pos[:, idx].reshape(n * k, 5, 3), quat[:, idx].reshape(n * k, 5, 4))
        self.windows = self.reference.frames.reshape(n, k, -1)
        self.features = human_features_batch(self.windows, self.mask).astype(np.float32)
        if self.graph is not None:
            self.device_features = torch.as_tensor(self.features, device=self.graph.device)

    def step(self, imu, gyro, q, dq):
        """Native joint targets [N, J] for IMU orientations [N, 4], gyros [N, 3] and joints [N, J]."""
        robot, n = self.robot, self.n
        quat = imu / np.linalg.norm(imu, axis=-1, keepdims=True)
        gravity = np_rotate_back(quat, np.array([0.0, 0.0, -1.0]))
        frame = np.concatenate([robot.to_canonical(q) - self.default_canonical, robot.to_canonical_delta(dq),
                                gyro, gravity, self.last_action], -1).astype(np.float32)
        self.history = np.concatenate([self.history[:, 1:], frame[:, None]], 1)
        self.history[self.fresh] = frame[self.fresh, None]
        self.fresh[:] = False
        torso = anchor_quat_batch(robot, quat, q)
        feeds = {"obs_history": self.history.reshape(n, -1),
                 "human_point_mask": np.broadcast_to(self.mask, (n, len(self.mask))).copy(),
                 "human_frame_mask": np.ones((n, GOAL_K), bool),
                 "human_heading": heading_features_batch(torso, self.windows, self.mask).astype(np.float32),
                 "human_heading_mask": np.ones(n, bool)}
        if self.graph is None:
            feeds.update(self.runner.inputs, goal_human=self.features)
            action = self.runner.session.run(["action_human"], feeds)[0].astype(np.float64)
        else:
            device = self.graph.device
            feeds = {k: torch.as_tensor(v, device=device) for k, v in feeds.items()}
            feeds.update(self.constants, goal_human=self.device_features)
            action = self.graph.run(["action_human"], feeds)[0].double().cpu().numpy()
        self.last_action = action
        delta = np.zeros((n, robot.num_native))
        delta[:, robot.perm] = robot.sign * action * robot.action_scale[robot.perm]
        return robot.default_joint_pos + delta


class BatchedOfficialFive(DiffusionObservations):
    def __init__(self, env, fm_path: Path, release: Path = Path("downloads/morphmimic-v0.1"), fm_batch: int = 256,
                 tracker_device=None, gripper_hysteresis=(0.02, 0.05), wrist_height_bias=None):
        self.env = env
        # The v0.1 tracker's steady wrist height error grows with how high the hand is held:
        # error ~ c0 + c1 * h (m), h the target wrist height above the target pelvis. With (c0, c1)
        # the tracker receives wrist targets lowered by that error, i.e. raised when c1 h dominates.
        # The FM plan itself (plan_pos) is unchanged. None: targets as planned.
        self.wrist_height_bias = wrist_height_bias
        self.fm_batch = fm_batch  # worlds per FM forward pass; bounds peak memory
        self.fm, config = load_policy(fm_path, device=env.device)
        ds = config["dataset"]
        if (self.fm.action_dim != 47 or tuple(ds["feature_names"]["observation.state"]) != C.BODY_JOINT_NAMES
                or tuple(FM_POINTS) != ("pelvis", "left_ankle", "right_ankle", "left_wrist", "right_wrist")
                or self.fm.horizon < self.fm.execute_steps + PREVIEW_FRAMES - 1):
            raise ValueError("Incompatible FM, preview horizon or G1 joint/keypoint layout")
        super().__init__(env, self.fm.obs_horizon, ds["image_size"], "head_camera")
        self.n = env.num_envs
        self.tracker = BatchedHumanFive(release, self.n, self.fm.execute_steps, self.fm.horizon,
                                        device=tracker_device or env.device)
        robot = self.tracker.robot
        self.action_term = env.action_manager.get_term("joint_pos")
        if robot.native_joint_names != C.BODY_JOINT_NAMES or tuple(self.action_term.target_names) != C.BODY_JOINT_NAMES:
            raise ValueError("Official G1 joint order differs from the scene action order")
        if not np.isclose(env.step_dt, robot.control_dt):
            raise ValueError("Scene and ONNX policy control rates differ")
        # The FM was trained under BF16 autocast (config precision: bf16); infer the same way.
        self.precision = torch.autocast(device_type=torch.device(env.device).type, dtype=torch.bfloat16)
        # Only the last obs_horizon frames of a plan reach the next decision: other control steps
        # may skip camera rendering (contacts and dynamics do not depend on it).
        self.render = True
        sense = env.sim.sense

        def gated_sense():
            if self.render:
                sense()
        env.sim.sense = gated_sense
        # The demonstrations command the right gripper fully open or closed only; the FM regresses
        # that binary command and drifts between the two while carrying. With hysteresis the
        # right gripper closes below the first width (m), reopens above the second and otherwise
        # keeps its state; the command is then the binary one of the data. None: raw FM widths.
        self.gripper_hysteresis = gripper_hysteresis
        self.grip_closed = torch.zeros(self.n, dtype=torch.bool, device=env.device)
        self.commanded_width = None
        self.nonfinite = torch.zeros(self.n, dtype=torch.bool, device=env.device)
        self.bad_plan = torch.zeros(self.n, dtype=torch.bool, device=env.device)
        self.offset = 0
        self.plan_pos = None
        print(f"Loaded {fm_path.name} and official v0.1 action_human (g1_amber, five slots) for {self.n} worlds",
              flush=True)

    # ------------------------------------------------------------------ episodes

    @torch.no_grad()
    def reset_worlds(self, ids: torch.Tensor):
        """Start the tracker state and FM observation history of worlds ``ids`` afresh."""
        self.tracker.reset(ids.cpu().numpy())
        # A start with the fingers on the can (measured width below fully open) is a closed gripper.
        measured = gripper_opening(self.robot.data.joint_pos[:, self.gripper_ids])[:, 1]
        self.grip_closed[ids] = measured[ids] < C.GRIPPER_OPEN_WIDTH - 0.005
        image, state = self.diffusion_image(), self.diffusion_state()
        if self.history is None:
            self.history = (image[:, None].repeat(1, self.obs_horizon, 1, 1, 1),
                            state[:, None].repeat(1, self.obs_horizon, 1))
        else:
            # Only the reset worlds: the others keep their frames (no control step has passed).
            images, states = self.history
            images[ids] = image[ids, None]
            states[ids] = state[ids, None]

    def push_history(self):
        """Append this control step's frame for every world."""
        return self._push_history(torch.zeros(self.n, dtype=torch.bool, device=self.env.device))

    # ------------------------------------------------------------------ planning

    @torch.no_grad()
    def encode(self, ids=None):
        images, states = self.history
        if ids is not None:
            images, states = images[ids], states[ids]
        with self.precision:
            return torch.cat([self.fm.encode(images[i:i + self.fm_batch], states[i:i + self.fm_batch]).float()
                              for i in range(0, len(images), self.fm_batch)])

    @torch.no_grad()
    def sample(self, cond, noise):
        with self.precision:
            parts = [self.fm.sample(cond[i:i + self.fm_batch], noise=noise[i:i + self.fm_batch])
                     for i in range(0, len(cond), self.fm_batch)]
        return {k: torch.cat([part[k] for part in parts]) for k in parts[0]}

    @torch.no_grad()
    def set_plan(self, prediction):
        data = self.robot.data
        pred, grip = prediction["body_action_pred"], prediction["gripper_width_pred"]
        pelvis_pos, pelvis_quat = data.body_link_pos_w[:, self.pelvis], data.body_link_quat_w[:, self.pelvis]
        # Worlds with a nonfinite plan or pelvis pose plan from finite stand-ins; execute_plan
        # flags them so the caller ends their episodes.
        self.bad_plan = ~(torch.isfinite(pred).flatten(1).all(-1) & torch.isfinite(grip).flatten(1).all(-1)
                          & torch.isfinite(pelvis_pos).all(-1) & torch.isfinite(pelvis_quat).all(-1)
                          & (pelvis_quat.norm(dim=-1) > 1e-8))
        if self.bad_plan.any():
            b = self.bad_plan
            pred = torch.where(b[:, None, None], torch.zeros_like(pred), pred)
            grip = torch.where(b[:, None, None], torch.zeros_like(grip), grip)
            pelvis_pos = torch.where(b[:, None], torch.zeros_like(pelvis_pos), pelvis_pos)
            pelvis_quat = torch.where(b[:, None], torch.tensor([1., 0., 0., 0.], device=pelvis_quat.device,
                                                               dtype=pelvis_quat.dtype), pelvis_quat)
        self.plan_pos, self.plan_quat = keypoints_to_world(pred, pelvis_pos, pelvis_quat)
        self.grip = grip
        self.offset = 0
        self.tracker.set_plan(self.tracker_targets().cpu().numpy().astype(np.float64),
                              self.plan_quat.cpu().numpy().astype(np.float64))

    def tracker_targets(self):
        """Five-point positions sent to the tracker: the plan, with the wrist height pre-compensation."""
        if self.wrist_height_bias is None:
            return self.plan_pos
        c0, c1 = self.wrist_height_bias
        pos = self.plan_pos.clone()
        wrists = pos[..., 3:5, 2]  # FM points: pelvis, ankles, left wrist, right wrist
        pos[..., 3:5, 2] = wrists - (c0 + c1 * (wrists - pos[..., :1, 2]))
        return pos

    # ------------------------------------------------------------------ control

    @torch.no_grad()
    def body_action(self):
        data = self.robot.data
        imu_t = data.body_link_quat_w[:, self.pelvis]
        gyro_t = quat_apply_inverse(imu_t, data.body_link_ang_vel_w[:, self.pelvis])
        imu, gyro = imu_t.cpu().numpy().astype(np.float64), gyro_t.cpu().numpy().astype(np.float64)
        q = data.joint_pos[:, self.body_ids].cpu().numpy().astype(np.float64)
        dq = data.joint_vel[:, self.body_ids].cpu().numpy().astype(np.float64)
        target = self.tracker.step(imu, gyro, q, dq)
        angles = torch.as_tensor(target, device=imu_t.device, dtype=imu_t.dtype)
        # Native target angles; mjlab applies its own scale and offset.
        return (angles - self.action_term.offset) / self.action_term.scale

    def gripper_width(self):
        """Left and right gripper width commands (m) of this control step."""
        width = self.grip[:, self.offset].clone()
        if self.gripper_hysteresis is not None:
            close, reopen = self.gripper_hysteresis
            right = width[:, 1]
            self.grip_closed = torch.where(right < close, True, torch.where(right > reopen, False, self.grip_closed))
            width[:, 1] = torch.where(self.grip_closed, 0.0, C.GRIPPER_OPEN_WIDTH)
        return width

    @torch.no_grad()
    def execute_plan(self):
        self.commanded_width = self.gripper_width()
        command = torch.cat((self.body_action(), gripper_q(self.commanded_width)), dim=-1)
        # A world with a nonfinite command (a NaN plan or state) gets a neutral command instead;
        # the caller ends its episode. One such world must not stop the other lockstep worlds.
        self.nonfinite = ~torch.isfinite(command).all(-1) | self.bad_plan
        if self.nonfinite.any():
            command = torch.where(self.nonfinite[:, None], torch.zeros_like(command), command)
        self.offset += 1
        return command
