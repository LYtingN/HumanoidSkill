"""phi: gripper and head-camera poses of 71-d body states (no parameters).

The poses only need the pelvis -> waist -> torso and torso -> arm -> wrist chains of SuperMimic's
G1 model (bare hands; 18 of its 30 bodies), so the forward kinematics runs over those bodies only
(as ``supermimic.kinematics.G1Kinematics``: each hinge rotates about its anchor by
``qpos - qpos0``), compiled into a few fused kernels. The ALOHA gripper bases sit rigidly on the
wrists and the head camera on the torso, at transforms read off this scene (the camera's from
its config, as ``env_cfg`` mounts it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from can_pnp.body.world_model import BodyWorldModel

SIDES = ("left", "right")


def _compose(pos, quat, offset_pos, offset_quat):
    return pos + quat_apply(quat, offset_pos.expand_as(pos)), quat_mul(quat, offset_quat.expand_as(quat))


class BodyKinematics:
    def __init__(self, env: ManagerBasedRlEnv, body: BodyWorldModel, compile: bool = True):
        fk = body.fk
        targets = [fk.body_index(f"{s}_wrist_yaw_link") for s in SIDES] + [fk.body_index("torso_link")]
        needed: set[int] = set()
        for i in targets:
            while i >= 0 and i not in needed:
                needed.add(i)
                i = fk.links[i].parent_index
        self.order = sorted(needed)  # topological: MuJoCo bodies come after their parents
        self.slot = {b: k for k, b in enumerate(self.order)}
        self.links = [(self.slot.get(fk.links[b].parent_index, -1), fk.links[b].joint_type == mujoco.mjtJoint.mjJNT_HINGE,
                       fk.links[b].qpos_adr) for b in self.order]  # fmt: skip
        idx = torch.tensor(self.order, device=fk.device)
        self.body_pos, self.body_quat = fk.body_pos[idx], fk.body_quat[idx]
        self.jnt_pos, self.jnt_axis, self.qpos0 = fk.jnt_pos[idx], fk.jnt_axis[idx], fk.qpos0[idx]
        self.wrist = [self.slot[b] for b in targets[:2]]
        self.torso = self.slot[targets[2]]

        robot = env.scene["robot"]
        pos, quat = robot.data.body_link_pos_w[0], robot.data.body_link_quat_w[0]
        offsets = []
        for side in SIDES:
            w, g = (robot.body_names.index(f"{side}_{b}") for b in ("wrist_yaw_link", "gripper_base"))
            offsets.append((quat_apply_inverse(quat[w], pos[g] - pos[w]), quat_mul(quat_inv(quat[w]), quat[g])))
        self.gripper_pos = torch.stack([p for p, _ in offsets])  # (2, 3) in the wrist frame
        self.gripper_quat = torch.stack([q for _, q in offsets])  # (2, 4)
        camera = env.cfg.events["reset_layout"].params["scene_cfg"].camera
        self.camera_pos = torch.tensor(camera.pos, dtype=torch.float32, device=env.device)
        self.camera_quat = torch.tensor(camera.quat, dtype=torch.float32, device=env.device)
        self._poses = torch.compile(self._poses_eager, dynamic=None) if compile else self._poses_eager

    def _chain(self, s: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """World (env frame) positions / quaternions of the chain bodies, root first."""
        pos: list[torch.Tensor] = []
        quat: list[torch.Tensor] = []
        batch = s.shape[:-1]
        for k, (parent, hinge, qadr) in enumerate(self.links):
            if parent < 0:  # the free-joint root
                pos.append(s[..., 0:3])
                quat.append(s[..., 3:7] / s[..., 3:7].norm(dim=-1, keepdim=True))
                continue
            b_pos = pos[parent] + quat_apply(quat[parent], self.body_pos[k].expand(*batch, 3))
            b_quat = quat_mul(quat[parent], self.body_quat[k].expand(*batch, 4))
            if hinge:
                anchor = b_pos + quat_apply(b_quat, self.jnt_pos[k].expand(*batch, 3))
                turn = quat_from_angle_axis(s[..., qadr] - self.qpos0[k], self.jnt_axis[k].expand(*batch, 3))
                b_quat = quat_mul(b_quat, turn)
                b_pos = anchor - quat_apply(b_quat, self.jnt_pos[k].expand(*batch, 3))
            pos.append(b_pos)
            quat.append(b_quat)
        return pos, quat

    def _poses_eager(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        pos, quat = self._chain(s)
        torso_pos, torso_quat = pos[self.torso], quat[self.torso]
        wrist_pos = torch.stack([pos[i] for i in self.wrist], dim=-2)
        wrist_quat = torch.stack([quat[i] for i in self.wrist], dim=-2)
        gripper_pos, gripper_quat = _compose(wrist_pos, wrist_quat, self.gripper_pos, self.gripper_quat)
        camera_pos, camera_quat = _compose(torso_pos, torso_quat, self.camera_pos, self.camera_quat)
        return {
            "gripper_pos": gripper_pos, "gripper_quat": gripper_quat,
            "camera_pos": camera_pos, "camera_quat": camera_quat,
            "torso_pos": torso_pos, "torso_quat": torso_quat,
        }  # fmt: skip

    def poses(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        """States (..., 71) -> ``gripper_pos`` (..., 2, 3) / ``gripper_quat`` (..., 2, 4)
        (left, right), ``camera_pos`` (..., 3) / ``camera_quat`` (..., 4), ``torso_*`` alike;
        positions in the env frame of the states."""
        shape = s.shape[:-1]
        out = self._poses(s.reshape(-1, s.shape[-1]))
        return {k: v.reshape(*shape, *v.shape[1:]) for k, v in out.items()}
