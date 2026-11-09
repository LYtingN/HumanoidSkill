"""Imagined chunk returns of candidate noises, without rendering or physics.

From a recorded decision (observation tokens, robot state, tracker observation, can state and
layout) each noise is decoded by the frozen DiT into a reference, Phi (tracker + body world
model) executes the whole reference (``horizon`` frames), phi gives the right gripper poses, the
object world model rolls the can on them and the commanded openings, and the task tracker of the
simulator (the supplied task tracker) scores the imagined can: the chunk return is the change of the
episode potential over the chunk (done > dropped (can below the drop height) > distance). With a
critic (``hskill.learners.critic``) the chunk-end potential gets the critic's remainder of the
episode added, unless the task ended inside the chunk. Body falls are not scored: Phi's
termination probability is only reported.

Every candidate is scored twice, on the same object world model: the actual branch moves the
object along the right-gripper path Phi brings about, the ideal branch along the path of Phi's body
with the reference's own arm joints ("ideal only at the hands": the arms tracked exactly). The
first is what S0 achieves with the intent, the second what the intent is worth; their gap is the
value lost to arm tracking. With an ensemble object world model both come with the members'
disagreement, which flags hand paths the model has no data for.
"""

from __future__ import annotations

import torch

from hskill.embodiment import g1_aloha as C
from hskill.executors.imagine import PhiStep, imagine_body
from hskill.interaction.model import ObjectWMEnsemble, ObjectWorldModel
from hskill.learners.critic import (
    ProgressCritic,
    ValueCritic,
    critic_features,
    progress_features,
    state_proprio,
)
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
                 object_wm: ObjectWorldModel, task, geo, horizon: int = 32,
                 batch: int = 1024, critic: ValueCritic | ProgressCritic | None = None, *, task_tracker):  # fmt: skip
        self.source, self.phi, self.body, self.kin, self.critic = source, phi, phi.body, kin, critic
        self.object_wm, self.task, self.geo = object_wm, task, geo
        self.horizon, self.batch = horizon, batch
        self.task_tracker = task_tracker
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
        H, task = self.horizon, self.task
        joint_pos, grip, rot6d = self.source.decode(start["cond"], noise)
        s0 = start["s0"]
        n = len(s0)
        plan = reference_plan(
            joint_pos, rot6d, s0[:, 3:7], self.body.default_joint_pos.expand(n, -1), self.dt,
            self.legs, s0[:, 7:36], s0[:, 2],
        )  # fmt: skip
        body = imagine_body(self.phi, plan, s0, start["obs0"], start["xi"], H)
        command = grip[:, :H, 1].T  # (H, B) right opening, straight from the reference
        d0 = (start["obj0"][:, :3] - start["goal"]).norm(dim=-1)
        potential0 = torch.exp(-d0 / task.dist_scale)
        # Actual branch: the hands Phi brings about. Ideal branch: the same body with the reference's
        # arm joints (state k follows the target of control step k - 1), i.e. arms tracked exactly.
        actual = self._score(start, body.states, command)
        ideal_states = body.states.clone()
        ideal_states[1:, :, ARM_STATE] = plan.joint_pos[:, :H, ARM_JOINTS].transpose(0, 1)
        ideal = self._score(start, ideal_states, command)
        return {
            "returns": actual["final"] - potential0, "returns_ideal": ideal["final"] - potential0,
            # Chunk outcomes only (no critic), comparable with a simulated chunk.
            "chunk_returns": actual["chunk"] - potential0, "chunk_returns_ideal": ideal["chunk"] - potential0,
            "potential0": potential0, "value": actual["value"], "uncertainty": actual["spread"],
            "uncertainty_ideal": ideal["spread"], "placed": actual["placed"], "dropped": actual["dropped"],
            "final_dist": actual["final_dist"], "final_pos": actual["final_pos"], "fall_prob": body.fall_prob[-1],
        }  # fmt: skip

    def _score(self, start: dict[str, torch.Tensor], states: torch.Tensor, command: torch.Tensor) -> dict[str, torch.Tensor]:
        """Roll the object along the right-gripper path of body ``states`` (H + 1, B, 71) and score
        it with the task tracker (and the critic beyond the chunk)."""
        H, task, n = self.horizon, self.task, states.shape[1]
        poses = self.kin.poses(states)
        gripper = torch.cat([poses["gripper_pos"][..., 1, :], poses["gripper_quat"][..., 1, :]], dim=-1)
        args = (start["obj0"], gripper, command, start["box"], start["table"])
        if isinstance(self.object_wm, ObjectWMEnsemble):
            all_objs, _ = self.object_wm.rollout_all(*args)
            objs, spread = ObjectWMEnsemble.mean(all_objs), ObjectWMEnsemble.spread(all_objs)
        else:
            objs, _ = self.object_wm.rollout(*args)
            spread = torch.zeros(n, device=states.device)
        zeros = torch.zeros(n, device=states.device)
        tracker = self.task_tracker(task, self.geo, start["box"], start["goal"], zeros, start["obj0"][:, :3], self.dt)
        alive = torch.ones(n, dtype=torch.bool, device=states.device)
        dropped = torch.zeros_like(alive)
        for k in range(1, H + 1):
            pos, quat = objs[k, :, :3], objs[k, :, 3:7]
            drop = alive & (pos[:, 2] < task.drop_height)
            dropped |= drop
            tracker.step(pos, quat, command[k - 1] > task.release_opening, alive & ~drop)
            alive &= ~(tracker.placed | drop)
        final = tracker.potential(dropped)
        chunk = final  # the chunk's own outcome, before the critic: what the simulator can check
        value = torch.full_like(final, float("nan"))
        if self.critic is not None:
            # Episode return from the chunk end: done / dropped inside the chunk, else the critic's.
            time = (start["time"] + H * start["step_frac"]).clamp(max=1.0)
            if isinstance(self.critic, ProgressCritic):
                # Demonstration progress of the chunk end's body (no object part: see ProgressCritic).
                feat = progress_features(state_proprio(self.kin, states[-1], objs[-1][:, 13]), None)
                value = self.critic(feat).clamp(0.0, task.success_reward)
            else:
                feat = critic_features(self.body, states[-1], objs[-1], start["box"], start["table"], start["goal"], time)
                value = (final + self.critic(feat)).clamp(0.0, task.success_reward)
            final = torch.where(tracker.placed | dropped, final, value)
        return {"final": final, "chunk": chunk, "value": value, "spread": spread, "placed": tracker.placed, "dropped": dropped,
                "final_dist": tracker.final_dist, "final_pos": objs[-1, :, :3]}  # fmt: skip
