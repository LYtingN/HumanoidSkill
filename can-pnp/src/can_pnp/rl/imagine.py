"""Imagined chunk returns of candidate noises, without rendering or physics.

From a recorded decision (observation tokens, robot state, tracker observation, can state and
layout) each noise is decoded by the frozen DiT into a reference, Phi (tracker + body world
model) executes the whole reference (``horizon`` frames), phi gives the right gripper poses, the
object world model rolls the can on them and the commanded openings, and the task tracker of the
simulator (``can_pnp.task``) scores the imagined can: the chunk return is the change of the
episode potential over the chunk (done > dropped (can below the drop height) > distance). With a
critic (``can_pnp.rl.critic``) the chunk-end potential gets the critic's remainder of the
episode added, unless the task ended inside the chunk. Body falls are not scored: Phi's
termination probability is only reported.
"""

from __future__ import annotations

import torch

from can_pnp import constants as C
from can_pnp.body.imagine import PhiStep, imagine_body
from can_pnp.body.kinematics import BodyKinematics
from can_pnp.config import SceneGeometryCfg
from can_pnp.dsrl import SteeredDiffusionSource
from can_pnp.object_wm.model import ObjectWorldModel
from can_pnp.odometry import LegKinematics
from can_pnp.pipeline import reference_plan
from can_pnp.rl.critic import ValueCritic, critic_features
from can_pnp.task import TaskCfg, TaskTracker


class Imaginer:
    def __init__(self, source: SteeredDiffusionSource, phi: PhiStep, kin: BodyKinematics,
                 object_wm: ObjectWorldModel, task: TaskCfg, geo: SceneGeometryCfg, horizon: int = 32,
                 batch: int = 1024, critic: ValueCritic | None = None):  # fmt: skip
        self.source, self.phi, self.body, self.kin, self.critic = source, phi, phi.body, kin, critic
        self.object_wm, self.task, self.geo = object_wm, task, geo
        self.horizon, self.batch = horizon, batch
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
        poses = self.kin.poses(body.states)
        gripper = torch.cat([poses["gripper_pos"][..., 1, :], poses["gripper_quat"][..., 1, :]], dim=-1)
        command = grip[:, :H, 1].T  # (H, B) right opening, straight from the reference
        objs, _ = self.object_wm.rollout(start["obj0"], gripper, command, start["box"], start["table"])
        zeros = torch.zeros(n, device=s0.device)
        tracker = TaskTracker(task, self.geo, start["box"], start["goal"], zeros, start["obj0"][:, :3], self.dt)
        alive = torch.ones(n, dtype=torch.bool, device=s0.device)
        dropped = torch.zeros_like(alive)
        for k in range(1, H + 1):
            pos, quat = objs[k, :, :3], objs[k, :, 3:7]
            drop = alive & (pos[:, 2] < task.drop_height)
            dropped |= drop
            tracker.step(pos, quat, command[k - 1] > task.release_opening, alive & ~drop)
            alive &= ~(tracker.placed | drop)
        d0 = (start["obj0"][:, :3] - start["goal"]).norm(dim=-1)
        potential0 = torch.exp(-d0 / task.dist_scale)
        final = tracker.potential(dropped)
        value = torch.full_like(final, float("nan"))
        if self.critic is not None:
            # Episode return from the chunk end: done / dropped inside the chunk, else the critic's.
            time = (start["time"] + H * start["step_frac"]).clamp(max=1.0)
            feat = critic_features(self.body, body.states[-1], objs[-1], start["box"], start["table"], start["goal"], time)
            value = (final + self.critic(feat)).clamp(0.0, task.success_reward)
            final = torch.where(tracker.placed | dropped, final, value)
        return {
            "returns": final - potential0, "potential0": potential0, "value": value,
            "placed": tracker.placed, "dropped": dropped,
            "final_dist": tracker.final_dist, "final_pos": objs[-1, :, :3], "fall_prob": body.fall_prob[-1],
        }  # fmt: skip
