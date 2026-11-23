"""Five-point FM planner and the MorphMimic G1 keypoint controller, for one mjlab world."""
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from hskill.embodiment import g1_aloha as C
from hskill.s0.controller import DiffusionObservations, gripper_opening, gripper_q
from humanoid_diffusion.keypoints import KP_NAMES as FM_POINTS, keypoints_to_world
from humanoid_diffusion.model import load_policy


class FivePointPlanner(DiffusionObservations):
    preview_frames = 10

    def __init__(self, env, fm_path: Path):
        self.fm, config = load_policy(fm_path, device=env.device)
        ds = config["dataset"]
        if (env.num_envs != 1 or self.fm.action_dim != 47 or
                tuple(ds["feature_names"]["observation.state"]) != C.BODY_JOINT_NAMES or
                tuple(FM_POINTS) != ("pelvis", "left_ankle", "right_ankle", "left_wrist", "right_wrist") or
                self.fm.horizon < self.fm.execute_steps + self.preview_frames - 1):
            raise ValueError("Incompatible FM, preview horizon or G1 joint/keypoint layout")
        super().__init__(env, self.fm.obs_horizon, ds["image_size"], "head_camera")
        self.offset = 0
        self.plan_pos = None
        self.replans = 0
        self.noise_actor = None
        self.gripper_hysteresis = None  # (close below, reopen above) m, as in FlashSAC training
        self.grip_closed = None

    @torch.inference_mode()
    def act(self, reset: bool):
        images, states = self._push_history(torch.full((1,), reset, device=self.robot.data.joint_pos.device))
        if reset or self.grip_closed is None:
            measured = gripper_opening(self.robot.data.joint_pos[:, self.gripper_ids])[:, 1]
            self.grip_closed = measured < C.GRIPPER_OPEN_WIDTH - 0.005
        if reset or self.plan_pos is None or self.offset == self.fm.execute_steps:
            if self.noise_actor is None:
                prediction = self.fm.predict_action(images, states)
            else:
                # As in FlashSAC training: the FM runs under BF16 autocast, its training precision.
                with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
                    cond = self.fm.encode(images, states).float()
                mean, _ = self.noise_actor.get_mean_and_std(cond.flatten(1), training=False)
                # A repeated-noise actor outputs one frame of latent, shared by every frame of the chunk.
                frames = 1 if self.noise_repeat else self.fm.horizon
                noise = (self.noise_scale * mean.tanh().reshape(1, frames, self.fm.action_dim)).expand(
                    1, self.fm.horizon, self.fm.action_dim)
                with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
                    prediction = self.fm.sample(cond, noise=noise)
            self.set_prediction(prediction)
        return self.execute_plan(reset)

    def load_flashsac_actor(self, checkpoint):
        from hskill.learners.flashsac.network import FlashSACActor

        device = self.fm.obs_center.device
        weights = torch.load(checkpoint, map_location=device, weights_only=True)
        if weights.get("algorithm") != "flashsac":
            raise ValueError("Expected a five-point FlashSAC checkpoint")
        cfg = weights["config"]
        repeat = bool(weights.get("noise_repeat", False))
        if (cfg["obs_dim"], cfg["action_dim"]) != (self.fm.n_tokens * self.fm.n_emb,
                                                   (1 if repeat else self.fm.horizon) * self.fm.action_dim):
            raise ValueError("FlashSAC checkpoint and FM dimensions differ")
        actor = FlashSACActor(cfg["blocks"], cfg["obs_dim"], cfg["actor_width"], cfg["action_dim"],
                              cfg.get("prior_std"))
        actor.load_state_dict(weights["actor"])
        self.noise_actor = actor.to(device).eval().requires_grad_(False)
        self.noise_scale = weights["noise_scale"]
        self.noise_repeat = repeat
        hysteresis = weights.get("gripper_hysteresis")
        self.gripper_hysteresis = tuple(hysteresis) if hysteresis else None
        return weights.get("episode_seconds", 30.)

    def set_prediction(self, prediction):
        data = self.robot.data
        self.plan_pos, self.plan_quat = keypoints_to_world(
            prediction["body_action_pred"], data.body_link_pos_w[:, self.pelvis],
            data.body_link_quat_w[:, self.pelvis],
        )
        self.grip = prediction["gripper_width_pred"]
        self.offset = 0
        self.replans += 1

    def execute_plan(self, reset):
        window = slice(self.offset, self.offset + self.preview_frames)
        body = self.body_action(reset, window)
        width = self.grip[:, self.offset].clone()
        if self.gripper_hysteresis is not None:
            close, reopen = self.gripper_hysteresis
            right = width[:, 1]
            self.grip_closed = torch.where(right < close, True, torch.where(right > reopen, False, self.grip_closed))
            width[:, 1] = torch.where(self.grip_closed, 0.0, C.GRIPPER_OPEN_WIDTH)
        command = torch.cat((body, gripper_q(width)), dim=-1)
        if not torch.isfinite(command).all():
            raise FloatingPointError("Nonfinite MorphMimic or gripper command")
        self.offset += 1
        return command


class FivePointController(FivePointPlanner):
    def __init__(self, env, fm_path: Path, tracker_path: Path):
        from morphmimic import constants as M
        from morphmimic.model.keypoints import fixed_kp_masks
        from morphmimic.model.policy import Policy, PolicyCfg
        from morphmimic.train.normalizers import RunningNormalizer
        from morphmimic.train.obs_history import ObsHistory

        super().__init__(env, fm_path)
        if (tuple(FM_POINTS) != M.KP_NAMES or M.JOINT_NAMES != C.BODY_JOINT_NAMES
                or M.GOAL_K != self.preview_frames):
            raise ValueError("Incompatible MorphMimic joint, keypoint or preview layout")
        ckpt = torch.load(tracker_path, map_location="cpu", weights_only=False, mmap=True)
        cfg = ckpt["cfg"]
        if (cfg["robot"], cfg["kp_interface"], cfg["kp_masking"], ckpt["kp_goal_version"],
                ckpt["imu_body"]) != ("g1", True, "full", M.KP_GOAL_VERSION, "pelvis"):
            raise ValueError("Expected the G1 full-mask, pelvis-IMU keypoint tracker")
        training_pose = ckpt["wm"]["default_joint_pos"].reshape(-1)
        torch.testing.assert_close(self.robot.data.default_joint_pos[0, self.body_ids].cpu(), training_pose)
        self.tracker = Policy(PolicyCfg(
            enc_hidden=cfg["policy_hidden"], dec_hidden=cfg["policy_hidden"], act=cfg["act"],
            hist_len=cfg["hist_len"], keypoint=True, bf16=cfg["bf16"],
            morph_dim=ckpt["policy"]["decoder.morph_actor"].numel(), film=cfg["decoder_film"],
        ))
        self.tracker.load_state_dict(ckpt["policy"])
        self.tracker.to(env.device).eval()
        self.obs_norm = RunningNormalizer(M.PROPRIO_DIM)
        self.kp_norm = RunningNormalizer(M.KP_FEAT_DIM)
        for norm, key in ((self.obs_norm, "obs_norm"), (self.kp_norm, "kp_norm")):
            norm.load_state_dict(ckpt[key])
            norm.to(env.device).eval()
        self.proprio = ObsHistory(1, cfg["hist_len"], M.PROPRIO_DIM, env.device)
        self.tracked = [self.robot.body_names.index(name) for name in M.TRACKED_BODY_NAMES]
        self.masks = fixed_kp_masks(1, "full", M.GOAL_K, torch.device(env.device))
        self.previous = torch.zeros((1, M.ACTION_DIM), device=env.device)
        print(f"Loaded {fm_path.name} and {tracker_path.name}, tracker iteration {ckpt['iteration']}")

    def body_action(self, reset, window):
        from morphmimic.model.keypoints import mask_keypoint_goal, targets_features
        from morphmimic.model.state import StateParts, join_state
        from morphmimic.train.normalizers import normalize_history

        data = self.robot.data
        q, qd = data.joint_pos[:, self.body_ids], data.joint_vel[:, self.body_ids]
        tracked = self.tracked
        state = join_state(StateParts(q, qd, data.body_link_pos_w[:, tracked],
                                      data.body_link_quat_w[:, tracked], data.body_link_lin_vel_w[:, tracked],
                                      data.body_link_ang_vel_w[:, tracked]))
        feats = targets_features(state, self.plan_pos[:, window], self.plan_quat[:, window])
        goal = mask_keypoint_goal(self.kp_norm(feats), *self.masks)
        quat = data.body_link_quat_w[:, self.pelvis]
        if reset:
            self.previous.zero_()
        frame = torch.cat((
            data.joint_pos_biased[:, self.body_ids] - data.default_joint_pos[:, self.body_ids], qd,
            quat_apply_inverse(quat, data.body_link_ang_vel_w[:, self.pelvis]),
            quat_apply_inverse(quat, data.gravity_vec_w), self.previous,
        ), dim=-1)
        if reset:
            self.proprio.reset(None, frame)
        else:
            self.proprio.push(frame)
        self.previous = self.tracker.forward_kp(normalize_history(self.obs_norm, self.proprio.flat()), goal)
        return self.previous
