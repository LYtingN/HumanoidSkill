"""Five-point FM -> MorphMimic v2; imports the checkpoint's original source.

Unlike SuperMimic, MorphMimic normalizes each 93D proprio frame and flattens
history TIME-major. Its 450D keypoint features use the *target* heading for
positions and the *robot's full pelvis orientation* for orientations.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from hskill.embodiment import g1_aloha as C
from hskill.s0.controller import SuperMimicController, gripper_q


class KeypointTracker:
    goal_dim = 450

    def __init__(self, checkpoint: str | Path, device: str):
        from morphmimic.constants import KP_GOAL_VERSION
        from morphmimic.model.policy import Policy, PolicyCfg
        from morphmimic.train.normalizers import RunningNormalizer

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        if (ckpt.get("kp_goal_version") != KP_GOAL_VERSION or KP_GOAL_VERSION != 2
                or ckpt.get("imu_body") != "pelvis" or cfg.get("robot") != "g1"
                or cfg.get("kp_masking") != "full"):
            raise ValueError("Expected G1 pelvis-IMU, v2, full-five-point MorphMimic checkpoint")
        hidden = tuple(cfg["policy_hidden"])
        self.policy = Policy(PolicyCfg(enc_hidden=hidden, dec_hidden=hidden,
                                      hist_len=cfg["hist_len"], act=cfg["act"],
                                      keypoint=True, bf16=cfg["bf16"] and device.startswith("cuda")))
        self.policy.load_state_dict(ckpt["policy"], strict=True)
        self.policy.to(device).eval()
        self.obs_norm, self.kp_norm = RunningNormalizer(93), RunningNormalizer(450)
        for name in ("obs_norm", "kp_norm"):
            norm = getattr(self, name)
            norm.load_state_dict(ckpt[name], strict=True)
            norm.to(device).eval()
        self.hist_len = cfg["hist_len"]

    @torch.inference_mode()
    def __call__(self, history, state, pos_w, quat_w):
        from morphmimic.model.keypoints import targets_features, mask_keypoint_goal, fixed_kp_masks
        from morphmimic.train.normalizers import normalize_history

        features = targets_features(state, pos_w, quat_w)
        points, frames = fixed_kp_masks(state.shape[0], "full", 10, state.device)
        goal = mask_keypoint_goal(self.kp_norm(features), points, frames)
        return self.policy.forward_kp(normalize_history(self.obs_norm, history), goal)


class KeypointPlan:
    """World poses frozen at the FM prediction anchor; consume a sliding 10-frame window."""

    def __init__(self, prediction, pelvis_pos, pelvis_quat):
        from humanoid_diffusion.keypoints import keypoints_to_world

        if prediction.ndim != 3 or prediction.shape[-1] != 47:
            raise ValueError("FM prediction must be (B,H,47)")
        self.pos_w, self.quat_w = keypoints_to_world(prediction[..., :45], pelvis_pos, pelvis_quat)
        self.gripper = prediction[..., 45:].clone()
        self.horizon = prediction.shape[1]

    def window(self, offset, length=10):
        offset = torch.as_tensor(offset, device=self.pos_w.device).expand(self.pos_w.shape[0])
        idx = offset[:, None] + torch.arange(length, device=offset.device)
        if bool((idx < 0).any()) or bool((idx >= self.horizon).any()):
            raise ValueError("Plan does not contain the requested tracker preview")
        rows = torch.arange(len(offset), device=offset.device)[:, None]
        return self.pos_w[rows, idx], self.quat_w[rows, idx]


class MorphMimicController(SuperMimicController):
    """Reuse the task's 48D observation/camera assembly, replace all tracker logic.

    FM replans every 8 simulation steps; its full 32-frame output supplies the
    tracker's next 10 frames even at offset 7. Reset rows discard both histories.
    """

    def __init__(self, env, policy, tracker):
        from morphmimic.constants import JOINT_NAMES, TRACKED_BODY_NAMES, KP_BODY_IDX
        from morphmimic.train.obs_history import ObsHistory

        if tuple(JOINT_NAMES) != C.BODY_JOINT_NAMES:
            raise ValueError("Body joint ordering differs from MorphMimic")
        if (policy.action_dim, policy.obs_dim, policy.obs_horizon) != (47, 48, 2):
            raise ValueError("Expected 47D action / 48D state / 2-frame FM checkpoint")
        if policy.horizon < policy.execute_steps + 9 or abs(env.step_dt - 0.02) > 1e-8:
            raise ValueError("Expected 50 Hz and enough full-horizon preview")
        source = SimpleNamespace(execute_steps=policy.execute_steps, obs_horizon=policy.obs_horizon)
        super().__init__(env, source, tracker)
        self.fm = policy
        self.tracked_ids = [self.robot.body_names.index(n) for n in TRACKED_BODY_NAMES]
        self.kp_ids = [self.tracked_ids[i] for i in KP_BODY_IDX]
        self.proprio_history = ObsHistory(env.num_envs, tracker.hist_len, 93, env.device)
        self.previous_action = torch.zeros(env.num_envs, 29, device=env.device)
        self.replans = 0

    def tracker_state(self):
        from morphmimic.model.state import StateParts, join_state

        d, ids = self.robot.data, self.tracked_ids
        return join_state(StateParts(d.joint_pos[:, self.body_ids], d.joint_vel[:, self.body_ids],
                                    d.body_link_pos_w[:, ids], d.body_link_quat_w[:, ids],
                                    d.body_link_lin_vel_w[:, ids], d.body_link_ang_vel_w[:, ids]))

    @torch.inference_mode()
    def act(self, obs, reset):
        del obs  # The old environment's 930D history is TERM-major; never feed it to MorphMimic.
        d = self.robot.data
        self.previous_action[reset] = 0
        self.step_in_chunk[reset] = 0
        quat = d.body_link_quat_w[:, self.pelvis]
        proprio = torch.cat((d.joint_pos[:, self.body_ids] - self.default_joint_pos,
                             d.joint_vel[:, self.body_ids],
                             quat_apply_inverse(quat, d.body_link_ang_vel_w[:, self.pelvis]),
                             quat_apply_inverse(quat, d.gravity_vec_w), self.previous_action), -1)
        self.proprio_history.push(proprio)
        self.proprio_history.reset(reset.nonzero().flatten(), proprio[reset])
        images, states = self._push_history(reset)
        replan = (self.step_in_chunk >= self.execute_steps) | reset
        if self.plan is None:
            replan[:] = True
        if bool(replan.any()):
            pred = self.fm.predict_action(images[replan], states[replan])["action_pred"]
            plan = KeypointPlan(pred, d.body_link_pos_w[replan, self.pelvis], quat[replan])
            if self.plan is None:
                self.plan = plan
            else:
                for name in ("pos_w", "quat_w", "gripper"):
                    getattr(self.plan, name)[replan] = getattr(plan, name)
            self.step_in_chunk[replan] = 0
            self.replans += 1
        offset = self.step_in_chunk.clone()
        pos, quat = self.plan.window(offset)
        body = self.tracker(self.proprio_history.flat(), self.tracker_state(), pos, quat)
        grip = self.plan.gripper[torch.arange(self.env.num_envs, device=self.env.device), offset]
        action = torch.cat((body, gripper_q(grip)), -1)
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("Nonfinite policy action; simulation stopped")
        self.previous_action = body.clone()
        self.step_in_chunk += 1
        return action
