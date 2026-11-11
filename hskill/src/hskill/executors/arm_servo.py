"""Strong-servo arms: the simulated ideal executor E* of the cross-layer attribution.

In the envs of ``mask`` the arm joints stop following the tracker and servo the reference
itself with stiff PD gains, while legs, waist and balance stay with SuperMimic and the can,
table and box keep their physics. The difference to the other envs (same reference, real S0)
is then the task value lost to the tracker's arm tracking error ("ideal only at the hands").

The servo is a stiffer copy of the arm actuators (kp x ``kp_scale``, kd x sqrt(kp_scale), so
the damping ratio stays; force limits x ``effort_scale``), not a kinematic override: contacts
stay physical and the arm cannot push through the table.

The arm targets replace the tracker's in the processed actions of the ``joint_pos`` term only;
its raw action, which the tracker observes as its last action, stays the tracker's own. Each
step servos to the next reference frame (frame 0 of a plan is the state at the replan) with a
velocity feed-forward: a position actuator ``kp (q* - q) - kd qd`` with
``q* = q_ref + kd / kp qd_ref`` is ``kp (q_ref - q) + kd (qd_ref - qd)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from hskill.embodiment import g1_aloha as C

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from hskill.s0.controller import SuperMimicController

ARM_JOINT_NAMES = tuple(n for n in C.BODY_JOINT_NAMES if any(p in n for p in ("shoulder", "elbow", "wrist")))
_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")


class ArmServo:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        controller: SuperMimicController,
        mask: torch.Tensor,
        kp_scale: float = 10.0,
        effort_scale: float = 4.0,
        joints: tuple[str, ...] = ARM_JOINT_NAMES,
        feedforward: bool = True,
    ):
        # ``controller`` may be None, attached later (``attach``): until then nothing is servoed.
        self.env, self.controller = env, controller
        self.mask = mask.to(env.device, dtype=torch.bool)
        self.term = env.action_manager.get_term("joint_pos")
        self.cols = torch.tensor([self.term.target_names.index(n) for n in joints], device=env.device)
        # Reference columns: the plan holds the 29 body joints in C.BODY_JOINT_NAMES order.
        self.ref_cols = torch.tensor([C.BODY_JOINT_NAMES.index(n) for n in joints], device=env.device)

        ctrl, names = [], []
        for act in env.scene["robot"].actuators:
            for i, name in enumerate(act.target_names):
                if name in joints:
                    ctrl.append(int(act.global_ctrl_ids[i]))
                    names.append(name)
        if sorted(names) != sorted(joints):
            raise ValueError(f"Arm joints without a single-element position actuator: {set(joints) - set(names)}")
        order = [names.index(n) for n in joints]
        self.ctrl_ids = torch.tensor(ctrl, device=env.device)[order]

        sim = env.sim
        missing = tuple(f for f in _FIELDS if f not in sim._expanded_fields)
        sim.expand_model_fields(missing)
        kp = sim.get_default_field("actuator_gainprm")[self.ctrl_ids, 0]
        kd = -sim.get_default_field("actuator_biasprm")[self.ctrl_ids, 2]
        force = sim.get_default_field("actuator_forcerange")[self.ctrl_ids]
        self.default = (kp, kd, force)
        self.kp, self.kd, self.force = kp * kp_scale, kd * math.sqrt(kp_scale), force * effort_scale
        self.set_active(self.mask.nonzero().flatten(), True)
        self.lead = self.kd / self.kp if feedforward else torch.zeros_like(self.kp)

        self._process = self.term.process_actions
        self.term.process_actions = self._process_actions

    def attach(self, controller: SuperMimicController | None) -> None:
        """Servo the plans of ``controller`` (None: none)."""
        self.controller = controller
        self.active = controller is not None and bool(self.mask.any())

    def set_active(self, envs: torch.Tensor, active: bool) -> None:
        """Hand the arms of ``envs`` to the stiff servo (``active``) or back to the tracker with the
        actuators' own gains (e.g. to switch executors mid-episode)."""
        kp, kd, force = (self.kp, self.kd, self.force) if active else self.default
        rows, cols = envs[:, None], self.ctrl_ids
        model = self.env.sim.model
        model.actuator_gainprm[rows, cols, 0] = kp
        model.actuator_biasprm[rows, cols, 1] = -kp
        model.actuator_biasprm[rows, cols, 2] = -kd
        model.actuator_forcerange[rows, cols] = force
        self.mask[envs] = active
        self.active = self.controller is not None and bool(self.mask.any())

    def reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Arm joint position / velocity (N, J) of the reference frame this step servos to."""
        c = self.controller
        n = self.env.num_envs
        # act() has advanced step_in_chunk past the step being executed.
        k = torch.clamp(c.step_in_chunk, max=c.plan.horizon - 1)
        rows = torch.arange(n, device=self.env.device)
        return c.plan.joint_pos[rows, k][:, self.ref_cols], c.plan.joint_vel[rows, k][:, self.ref_cols]

    def _process_actions(self, actions: torch.Tensor) -> None:
        self._process(actions)
        if not self.active:  # no env servoed (or no controller yet): the tracker's targets stand
            return
        q, qd = self.reference()
        target = q + self.lead * qd
        processed = self.term._processed_actions
        processed[:, self.cols] = torch.where(self.mask[:, None], target, processed[:, self.cols])

    def tracking_error(self) -> torch.Tensor:
        """(N,) RMS arm joint error (rad) to the frame the last step servoed to."""
        q = self.env.scene["robot"].data.joint_pos[:, self.term.target_ids[self.cols]]
        return (q - self.reference()[0]).square().mean(-1).sqrt()

    def close(self) -> None:
        """Restore the action term (gains stay: ``set_active(envs, False)`` restores them)."""
        self.term.process_actions = self._process
