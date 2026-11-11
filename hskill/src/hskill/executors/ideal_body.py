"""Whole-body ideal executor E*: the reference executed (nearly) exactly in the simulator.

In the envs of ``mask`` every body joint servos the reference with stiff PD gains
(:class:`ArmServo` over all 29 body joints) and the root is held on the reference root at every
physics substep: its position by the plan's leg odometry from the root at the replan (as the ideal
branch of the imagination, ``reference_frames``), its rotation from the plan. ``root="wrench"``
pulls the pelvis there with a PD force and torque (a residual "hand of god" as in Residual Force
Control, Yuan & Kitani 2020); ``root="kinematic"`` writes the root state outright, which shakes a
held can loose (the root jumps under the contacts every substep). The joints, the grippers and the
can, table and box keep their physics: the arms cannot push through the table and grasps are made
by the real gripper. The tracker still runs (its observations stay its own) but its body actions
are overridden.

Frames follow ArmServo: the control step with plan offset ``k`` servos to frame ``k + 1`` (frame 0
is the state at the replan); the root moves linearly from frame ``k`` to ``k + 1`` over the substeps.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix, quat_inv, quat_mul

from hskill.embodiment import g1_aloha as C
from hskill.executors.arm_servo import ArmServo

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from hskill.s0.controller import SuperMimicController


class RootPin:
    """Holds the envs of ``mask`` on the reference root before every physics substep.

    ``mode="wrench"``: a PD wrench on the pelvis, ``mass x (kp e + kd de)`` and
    ``inertia x (kp_rot e_rot + kd_rot de_rot)`` in the world frame (no gravity feed-forward: the
    legs carry the weight, the wrench takes the error). The damping acts on the light pelvis alone
    at high frequency, which an explicit force cannot do stably (NaN within steps): it is the free
    joint's ``dof_damping``, integrated implicitly by ``implicitfast``, and the wrench adds its
    reference-velocity feed-forward (gains stay in the env, as ArmServo's). ``mode="kinematic"``: the root state is
    written outright. :meth:`reference` is defined for every env (the tracker's envs get the root
    their reference asks for, from their own replan root), so it also serves as the target of
    their root error.
    """

    def __init__(self, env: ManagerBasedRlEnv, controller: SuperMimicController, mask: torch.Tensor,
                 mode: str = "wrench", kp: float = 1000.0, kp_rot: float = 100.0, inertia: float = 10.0):  # fmt: skip
        if mode not in ("wrench", "kinematic"):
            raise ValueError(f"mode must be 'wrench' or 'kinematic', got {mode!r}")
        self.env, self.controller, self.mode = env, controller, mode
        self.robot = env.scene["robot"]
        self.mask = mask.to(env.device, dtype=torch.bool)
        self.ids = self.mask.nonzero().flatten()
        index = self.robot.data.indexing
        self.q_adr, self.v_adr = index.free_joint_q_adr, index.free_joint_v_adr
        self.pelvis = [self.robot.body_names.index("pelvis")]
        self.mass = float(env.sim.mj_model.body_mass[index.body_ids.cpu().numpy()].sum())
        self.inertia = inertia  # kg m^2: one scalar for the whole body about the pelvis
        self.kp, self.kd = kp, 2.0 * math.sqrt(kp)  # 1/s^2, 1/s: critically damped
        self.kp_rot, self.kd_rot = kp_rot, 2.0 * math.sqrt(kp_rot)
        if mode == "wrench":
            sim = env.sim
            if "dof_damping" not in sim._expanded_fields:
                sim.expand_model_fields(("dof_damping",))
            self.damping = torch.tensor([self.mass * self.kd] * 3 + [self.inertia * self.kd_rot] * 3, device=env.device)
            self.set_active(self.ids, True)
        self.decimation = env.cfg.decimation
        self.substep = 0
        self.start = torch.zeros(env.num_envs, 2, device=env.device)  # root xy at the replan, env frame
        self.term = env.action_manager.get_term("joint_pos")
        self._apply = self.term.apply_actions
        self.term.apply_actions = self._apply_actions

    def attach(self, controller: SuperMimicController | None) -> None:
        """Hold the roots to the plans of ``controller`` (None: none; the odometry restarts at its replans)."""
        self.controller = controller

    def set_active(self, envs: torch.Tensor, active: bool) -> None:
        """Hold the root of ``envs`` (``active``) or let it go (e.g. to switch executors mid-episode)."""
        self.mask[envs] = active
        self.ids = self.mask.nonzero().flatten()
        if self.mode != "wrench":
            return
        damping = self.damping if active else torch.zeros_like(self.damping)
        self.env.sim.model.dof_damping[envs[:, None], self.v_adr.long()] = damping
        if not active and len(envs):
            zero = torch.zeros(len(envs), 1, 3, device=self.env.device)
            self.robot.write_external_wrench_to_sim(zero, zero, env_ids=envs, body_ids=self.pelvis)

    def reference(self, alpha: float = 1.0) -> tuple[torch.Tensor, ...]:
        """Root position (N, 3, env frame), quaternion (N, 4) and world linear / angular velocity
        (N, 3) at fraction ``alpha`` of the current control step's way from frame k to k + 1."""
        plan = self.controller.plan
        if not plan.full:
            raise ValueError("RootPin needs the full-observation tracker's root height and velocity")
        n = self.env.num_envs
        rows = torch.arange(n, device=self.env.device)
        # act() has advanced step_in_chunk past the step being executed (as ArmServo.reference).
        k = torch.clamp(self.controller.step_in_chunk, max=plan.horizon - 1)
        k0 = torch.clamp(k - 1, min=0)
        steps = torch.cat([torch.zeros_like(plan.root_lin_vel_w[:, :1, :2]), plan.root_lin_vel_w[:, :-1, :2] * plan.dt], 1)
        pos = torch.cat([self.start[:, None] + torch.cumsum(steps, 1), plan.root_height[..., None]], -1)
        p0, p1 = pos[rows, k0], pos[rows, k]
        q0, q1 = quat_from_matrix(plan.root_w[rows, k0]), quat_from_matrix(plan.root_w[rows, k])
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        quat = torch.nn.functional.normalize(q0 + alpha * (q1 - q0), dim=-1)
        ang = plan.root_w[rows, k] @ plan.ang_vel_b[rows, k][..., None]
        return p0 + alpha * (p1 - p0), quat, (p1 - p0) / plan.dt, ang[..., 0]

    def _apply_actions(self) -> None:
        self._apply()
        i = self.substep % self.decimation
        self.substep += 1
        if self.controller is None or self.controller.plan is None:  # nothing to hold the root to yet
            return
        if i == 0:
            # First substep of a control step: the derived root pose is fresh (forward() ran for
            # the observations). Envs that replanned in the last act() start their odometry here.
            fresh = self.controller.step_in_chunk == 1
            root = self.robot.data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
            self.start = torch.where(fresh[:, None], root, self.start)
        if len(self.ids) == 0:
            return
        pos, quat, lin, ang = self.reference((i + 1) / self.decimation)
        ids = self.ids
        if self.mode == "kinematic":
            state = torch.cat([pos + self.env.scene.env_origins, quat, lin, ang], -1)
            self.robot.write_root_state_to_sim(state[ids], env_ids=ids)
            return
        # Wrench: from the free joint's qpos, current at every substep.
        data = self.env.sim.data
        q = data.qpos[ids][:, self.q_adr]
        cur_quat = q[:, 3:7]
        err_pos = pos[ids] + self.env.scene.env_origins[ids] - q[:, :3]
        err_rot = axis_angle_from_quat(quat_mul(quat[ids], quat_inv(cur_quat)))
        # The damping's -kd v is the free joint's (implicit); + kd v_ref is fed forward here.
        force = self.mass * (self.kp * err_pos + self.kd * lin[ids])
        torque = self.inertia * (self.kp_rot * err_rot + self.kd_rot * ang[ids])
        self.robot.write_external_wrench_to_sim(force[:, None], torque[:, None], env_ids=ids, body_ids=self.pelvis)

    def close(self) -> None:
        self.term.apply_actions = self._apply
        self.set_active(self.ids, False)


def ideal_executor(env: ManagerBasedRlEnv, controller: SuperMimicController, mask: torch.Tensor,
                   kp_scale: float = 10.0, effort_scale: float = 4.0, root: str = "wrench") -> tuple[ArmServo, RootPin]:  # fmt: skip
    """The whole-body E* in the envs of ``mask``: all body joints servoed, the root held."""
    servo = ArmServo(env, controller, mask, kp_scale, effort_scale, joints=C.BODY_JOINT_NAMES)
    return servo, RootPin(env, controller, mask.clone(), root)
