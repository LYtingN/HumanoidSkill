"""Imagined chunk returns of candidate noises, without rendering or physics.

From a recorded decision (observation tokens, robot state, tracker observation, the object WM's
history of keypoints / openings / can, and the layout) each noise is decoded by the frozen DiT
into a reference, Phi (tracker + body world model) executes the whole reference (``horizon``
frames), phi gives its keypoints, the object world model predicts the can from the history and
those future keypoints with the commanded openings, and the task tracker of the
simulator (the supplied task tracker) scores the imagined can: the chunk return is the change of the
episode potential over the chunk (done > dropped (can below the drop height) > distance). With a
critic (``hskill.learners.critic``) and ``critic_weight`` > 0, a chunk that neither completes nor
drops is worth the critic's estimate of the episode's final potential from the chunk end instead
of the potential there (``critic_weight`` in [0, 1] blends the two), so the score sees beyond the
chunk; ``chunk_returns`` stay the chunk's own outcome. Body falls are not scored: Phi's
termination probability is only reported. The actual branch's window end (Phi's
body and keypoints, the predicted can and its held logit) is returned for the model-based critic
(``hskill.learners.critic``).

Every candidate is scored twice, on the same object world model and the same history; only the
future keypoints differ:

    J  = Psi( M(history, keypoints of Phi's body) )        actual
    J* = Psi( M(history, keypoints of the reference) )     ideal

The ideal branch is the intent executed exactly (``ideal="full"``): all joints and the root
rotation from the reference, the root position by the reference's leg odometry from the start, so
all five keypoints come from the reference. ``ideal="arms"`` keeps Phi's body and replaces only
the arm joints ("ideal only at the hands": pelvis and feet as Phi has them). The first branch is
what S0 achieves with the intent, the second what the intent is worth; their gap is the value lost
to execution. With an ensemble object world model both come with the members'
disagreement, which flags hand paths the model has no data for.
"""

from __future__ import annotations

from typing import Literal

import torch

from hskill.embodiment import g1_aloha as C
from hskill.executors.imagine import PhiStep, imagine_body, reference_frames
from hskill.interaction.model import ObjectWMEnsemble, ObjectWorldModel
from hskill.learners.critic import critic_features
from hskill.s0.controller import reference_plan
from hskill.s0.kinematics import BodyKinematics
from hskill.s0.odometry import LegKinematics
from hskill.s1.noise_policy import SteeredDiffusionSource

# Arm joints (left and right shoulder / elbow / wrist) in BODY_JOINT_NAMES and in the 71-d state.
ARM_JOINTS = slice(15, 29)
ARM_STATE = slice(7 + 15, 7 + 29)
assert all("shoulder" in j or "elbow" in j or "wrist" in j for j in C.BODY_JOINT_NAMES[ARM_JOINTS])


class Imaginer:
    def __init__(self, source: SteeredDiffusionSource, phi: PhiStep, kin: BodyKinematics,
                 object_wm: ObjectWorldModel | ObjectWMEnsemble, task, geo, horizon: int = 32,
                 batch: int = 1024, *, task_tracker, ideal: Literal["full", "arms"] = "full", critic=None,
                 critic_weight: float = 0.0):  # fmt: skip
        if ideal not in ("full", "arms"):
            raise ValueError(f"ideal must be 'full' or 'arms', got {ideal!r}")
        self.source, self.phi, self.body, self.kin, self.ideal = source, phi, phi.body, kin, ideal
        self.object_wm, self.task, self.geo = object_wm, task, geo
        self.horizon, self.batch = horizon, batch
        self.task_tracker = task_tracker
        self.critic, self.critic_weight = critic, critic_weight
        env = self.body.env
        self.legs = LegKinematics(env.sim.mj_model, C.BODY_JOINT_NAMES, env.device, prefix="robot/")
        self.dt = env.step_dt

    @torch.no_grad()
    def chunk_return(self, start: dict[str, torch.Tensor], noise: torch.Tensor) -> dict[str, torch.Tensor]:
        """``start`` fields (B, ...) as ``EpisodeBuffer.starts``, ``noise`` (B, dim)."""
        outs = [self._chunk({k: v[i : i + self.batch] for k, v in start.items()}, noise[i : i + self.batch])
                for i in range(0, len(noise), self.batch)]  # fmt: skip
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    def _chunk(self, start: dict[str, torch.Tensor], noise: torch.Tensor) -> dict[str, torch.Tensor]:
        H = self.horizon
        joint_pos, grip, rot6d = self.source.decode(start["cond"], noise)
        s0 = start["s0"]
        n = len(s0)
        plan = reference_plan(
            joint_pos, rot6d, s0[:, 3:7], self.body.default_joint_pos.expand(n, -1), self.dt,
            self.legs, s0[:, 7:36], s0[:, 2],
        )  # fmt: skip
        body = imagine_body(self.phi, plan, s0, start["obs0"], start["xi"], H)
        command = grip[:, :H].transpose(0, 1)  # (H, B, 2) both openings, straight from the reference
        # Actual branch: the keypoints Phi brings about. Ideal branch: the reference executed exactly
        # (state k is reference frame k - 1, the target of control step k - 1): the whole body, or
        # Phi's body with the reference's arm joints.
        actual_kp = self.kin.keypoints(body.states[1:])
        actual = self._score(start, actual_kp, command)
        if self.ideal == "full":
            ideal_states = reference_frames(plan, s0, H)
        else:
            ideal_states = body.states[1:].clone()
            ideal_states[..., ARM_STATE] = plan.joint_pos[:, :H, ARM_JOINTS].transpose(0, 1)
        ideal = self._score(start, self.kin.keypoints(ideal_states), command)
        potential0 = actual["phi0"]  # the task's shaping at the start (the same for both branches)
        end_ideal = ideal_states[-1]
        score, score_ideal = actual["final"], ideal["final"]
        value = torch.full_like(score, float("nan"))
        if self.critic is not None and self.critic_weight > 0:
            t_left = (1.0 - start["time"]) / start["step_frac"] * self.dt  # seconds of the episode left
            value = self._value(start, body.states[-1], actual_kp[-1], actual, t_left)
            value_ideal = self._value(start, end_ideal, self.kin.keypoints(end_ideal), ideal, t_left)
            score = score + self.critic_weight * (value - score)
            score_ideal = score_ideal + self.critic_weight * (value_ideal - score_ideal)
        return {
            "returns": score - potential0, "returns_ideal": score_ideal - potential0, "value": value,
            # Chunk outcomes, comparable with a simulated chunk (equal to the returns while no
            # value beyond the chunk is added).
            "chunk_returns": actual["chunk"] - potential0, "chunk_returns_ideal": ideal["chunk"] - potential0,
            "potential0": potential0, "uncertainty": actual["spread"],
            "uncertainty_ideal": ideal["spread"], "placed": actual["placed"], "dropped": actual["dropped"],
            "final_dist": actual["final_dist"], "final_pos": actual["final_pos"], "fall_prob": body.fall_prob[-1],
            "s_end": body.states[-1], "kp_end": actual_kp[-1], "obj_end": actual["obj_end"],
            "held_logit_end": actual["held_logit_end"],
        }  # fmt: skip

    def _value(self, start, s_end, kp_end, branch, t_left) -> torch.Tensor:
        """The episode's final potential from a branch's chunk end: the chunk's outcome once done (or
        dropped), else the critic's members' mean, within [0, success_reward]."""
        feat = critic_features(self.body, s_end, kp_end, branch["obj_end"], branch["held_logit_end"], start["goal"], t_left)
        v = self.critic(feat).mean(0).clamp(0.0, self.task.success_reward)
        return torch.where(branch["placed"] | branch["dropped"], branch["final"], v)

    def _score(self, start: dict[str, torch.Tensor], fut_kp: torch.Tensor, command: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict the object from the start's history and the future keypoints ``fut_kp``
        (H, B, 5, 7) with the commanded openings (H, B, 2), and score it with the task tracker."""
        H, task, n = self.horizon, self.task, fut_kp.shape[1]
        hist = [start[k].transpose(0, 1) for k in ("hist_kp", "hist_width", "hist_obj")]  # time-major
        all_objs, all_logits = self.object_wm.rollout_all(*hist, fut_kp, command)
        objs, spread = ObjectWMEnsemble.mean(all_objs), ObjectWMEnsemble.spread(all_objs)
        zeros = torch.zeros(n, device=fut_kp.device)
        tracker = self.task_tracker(task, self.geo, start["box"], start["goal"], zeros, start["obj0"][:, :3], self.dt)
        phi0 = tracker.shaping()
        alive = torch.ones(n, dtype=torch.bool, device=fut_kp.device)
        dropped = torch.zeros_like(alive)
        for k in range(1, H + 1):
            pos, quat = objs[k, :, :3], objs[k, :, 3:7]
            drop = alive & (pos[:, 2] < task.drop_height)
            dropped |= drop
            tracker.step(pos, quat, command[k - 1, :, 1] > task.release_opening, alive & ~drop)
            alive &= ~(tracker.placed | drop)
        final = tracker.potential(dropped)
        return {"final": final, "chunk": final, "phi0": phi0, "spread": spread, "placed": tracker.placed, "dropped": dropped,
                "final_dist": tracker.final_dist, "final_pos": objs[-1, :, :3], "obj_end": objs[-1],
                "held_logit_end": all_logits.mean(0)[-1]}  # fmt: skip
