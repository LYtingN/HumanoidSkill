"""Approximate G1-to-SMPL retargeting for the official v0.1 human/five tracker."""
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from deploy.sim2sim import human
from deploy.sim2sim.obs import GOAL_K, quat_apply, quat_apply_inverse as rotate_back, quat_inv, quat_mul, yaw_quat
from deploy.sim2sim.policy_node import PolicyRunner
from humanoid_diffusion.keypoints import KP_BODY_NAMES

from hskill.embodiment import g1_aloha as C
from hskill.s0.fivepoint import FivePointPlanner


class HumanFiveReference:
    """Use corresponding robot/human poses in the release's paired standing frame."""

    def __init__(self, stand_file, tracked_bodies, execute_steps, *, future_horizon=None):
        stand = json.loads(stand_file.read_text())["g1"]["state"]
        indices = [tracked_bodies.index(name) for name in KP_BODY_NAMES]
        self.robot_pos = np.asarray(stand[58:100]).reshape(14, 3)[indices]
        self.robot_quat = np.asarray(stand[100:156]).reshape(14, 4)[indices]
        self.slots = list(human.HUMAN_POINTS["five"])
        self.stand = human.human_stand().reshape(24, 7)
        human_pos = self.stand[self.slots, :3]
        self.human_rel = human_pos - human_pos[0]
        self.robot_rel = self.robot_pos - self.robot_pos[0]
        self.scale = np.ones(5)
        self.scale[1:] = (np.linalg.norm(self.human_rel[1:], axis=1) /
                          np.linalg.norm(self.robot_rel[1:], axis=1))
        self.height_ratio = human_pos[0, 2] / self.robot_pos[0, 2]
        self.axes = quat_mul(quat_inv(self.robot_quat), self.stand[self.slots, 3:])
        self.execute_steps = execute_steps
        self.future_indices = None
        if future_horizon is not None:
            if future_horizon - execute_steps < GOAL_K:
                raise ValueError("The FM plan needs at least 10 distinct frames after the execution prefix")
            self.future_indices = np.linspace(execute_steps, future_horizon - 1, GOAL_K).round().astype(int)
        self.frames = None

    def update(self, pos, quat):
        # In the paired stand's axes, scale each endpoint's displacement from rest.
        yaw_delta = quat_mul(yaw_quat(quat[:, :1]), quat_inv(yaw_quat(self.robot_quat[:1])))
        relative = rotate_back(yaw_delta, pos - pos[:, :1])
        human_rel = self.human_rel + (relative - self.robot_rel) * self.scale[:, None]
        pelvis = pos[:, :1].copy()
        pelvis[:, :, 2] *= self.height_ratio
        frames = np.broadcast_to(self.stand, (len(pos), 24, 7)).copy()
        frames[:, self.slots, :3] = pelvis + quat_apply(yaw_delta, human_rel)
        oriented = quat_mul(quat, self.axes)
        frames[:, self.slots, 3:] = oriented / np.linalg.norm(oriented, axis=-1, keepdims=True)
        self.frames = frames.reshape(len(pos), -1)

    def window(self, step_id, imu_quat, q):
        if self.future_indices is not None:
            return self.frames[self.future_indices]
        start = step_id % self.execute_steps
        return self.frames[start:start + GOAL_K]

    def reset(self):
        self.frames = None


class OfficialFiveController(FivePointPlanner):
    def __init__(self, env, fm_path, *, sample_future=True, release=Path("downloads/morphmimic-v0.1"),
                 g1_export=None):
        super().__init__(env, fm_path)
        export = release / "morphmimic_policy" if g1_export is None else g1_export
        robot_name = "g1_amber" if g1_export is None else "g1"
        robot = json.loads((export / f"{robot_name}.json").read_text())["robot"]
        self.reference = HumanFiveReference(release / "source/deploy/vr_teleop/gmr_configs/stand.json",
                                            robot["tracked_bodies"], self.fm.execute_steps,
                                            future_horizon=self.fm.horizon if sample_future else None)
        if sample_future:
            print(f"Tracker future frames (zero-based): {self.reference.future_indices.tolist()}", flush=True)
        self.runner = PolicyRunner(export, robot_name, self.reference, interface="human", human_points="five")
        self.action_term = env.action_manager.get_term("joint_pos")
        if (self.runner.robot.native_joint_names != C.BODY_JOINT_NAMES or
                tuple(self.action_term.target_names) != C.BODY_JOINT_NAMES):
            raise ValueError("Official G1 joint order differs from the scene action order")
        if not np.isclose(env.step_dt, self.runner.robot.control_dt):
            raise ValueError("Scene and ONNX policy control rates differ")
        self.step = 0
        print(f"Loaded {fm_path} and {export}/policy.onnx ({robot_name}, human/five)", flush=True)

    def body_action(self, reset, window):
        if reset:
            self.step = 0
            self.runner.reset()
            self.runner.prev_step = -1
        if self.offset == 0:
            self.reference.update(self.plan_pos[0].cpu().numpy(), self.plan_quat[0].cpu().numpy())
        data = self.robot.data
        imu = data.body_link_quat_w[:, self.pelvis]
        gyro = quat_apply_inverse(imu, data.body_link_ang_vel_w[:, self.pelvis])
        target = self.runner.step(self.step, imu[0].cpu().numpy(), gyro[0].cpu().numpy(),
                                  data.joint_pos[0, self.body_ids].cpu().numpy(),
                                  data.joint_vel[0, self.body_ids].cpu().numpy())
        self.step += 1
        angles = torch.as_tensor(target, device=imu.device, dtype=imu.dtype)[None]
        # PolicyRunner returns native target angles; mjlab applies its own scale and offset.
        return (angles - self.action_term.offset) / self.action_term.scale
