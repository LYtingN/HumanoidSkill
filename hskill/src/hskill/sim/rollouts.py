"""Batched grasp-start episodes of the steered diffusion policy in the simulator.

``args`` is any object with the fields of ``train_dsrl.Args`` used here (env / policy loading,
``episode_s``, the task configuration fields).
"""

from __future__ import annotations

import torch
from mjlab.utils.nan_guard import NanGuard

from hskill.s0.controller import SuperMimicController
from hskill.s0.supermimic import load_tracking_policy
from hskill.s1.noise_policy import NoisePolicy, SteeredDiffusionSource
from hskill.s1.sources import load_diffusion_policy


class Rollouts:
    """Env, frozen policies and grasp starts; :meth:`run` rolls one GRPO batch."""

    def __init__(self, args, device: str, rank: int = 0, num_envs: int | None = None, *,
                 env_factory, starts_factory, task_cfg, layout_term, apply_grasp_start, task_tracker,
                 val_episodes):
        self.args, self.device = args, device
        self.layout_term, self.apply_grasp_start, self.task_tracker = layout_term, apply_grasp_start, task_tracker
        n = num_envs or args.groups * args.group_size
        self.env = env_factory(n, args.seed + rank, self.device)
        self.diffusion, _ = load_diffusion_policy(args.diffusion_checkpoint, args.diffusion_repo, self.device)
        self.diffusion.requires_grad_(False)
        if args.compile:
            self.diffusion.compile_for_training()
        self.tracker = load_tracking_policy(args.tracker_checkpoint, device=self.device)
        self.starts = starts_factory(phase=args.grasp_phase)
        val = set(val_episodes)
        self.train_rows = [i for i, e in enumerate(self.starts.episode_ids) if e not in val]
        self.val_rows = [i for i, e in enumerate(self.starts.episode_ids) if e in val]
        self.sensors = self.env.sim._sensor_context
        self.geo = self.env.cfg.events["reset_layout"].params["scene_cfg"].geometry
        self.task = task_cfg(args)
        self.env.reset()

    def render(self, on: bool) -> None:
        # mjlab renders every env step in sim.sense(); without the context it skips camera work.
        self.env.sim._sensor_context = self.sensors if on else None

    def start(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        """Write grasp start ``rows`` (indices into ``self.starts``) into all envs; their obs."""
        env = self.env
        self.render(True)
        # Rollouts follow each other without env resets: restart the episode clock and the action
        # history (the tracker observes its last actions) before writing the grasp starts.
        env.episode_length_buf.zero_()
        env.action_manager.reset()
        # A world whose physics diverged in an earlier rollout keeps NaNs (e.g. in the solver warm
        # start) that writing a start state does not clear: reset it fully first.
        stale = NanGuard.detect_nans(env.sim.data)
        if bool(stale.any()):
            env.reset(env_ids=stale.nonzero().flatten())
        self.apply_grasp_start(env, torch.arange(env.num_envs, device=self.device), self.starts, rows)
        return env.obs_buf

    def tracker_for(self):
        env = self.env
        term = self.layout_term(env)
        return self.task_tracker(self.task, self.geo, term.box_center_w, term.goal_pos_w, env.scene.env_origins[:, 2],
                           env.scene["can"].data.root_link_pos_w, env.step_dt)  # fmt: skip

    def failures(self, alive: torch.Tensor, terminated: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(fell, dropped, diverged) among ``alive`` envs after an env step. Terminated envs are
        already reset: their causes come from the termination terms."""
        terms = self.env.termination_manager
        fell = alive & terms.get_term("fallen")
        dropped = alive & terms.get_term("can_dropped") & ~fell
        # Physics blow-up (NaN / Inf state): a failed episode like a fall; its state is unusable.
        diverged = alive & NanGuard.detect_nans(self.env.sim.data)
        return fell, dropped, diverged

    @torch.no_grad()
    def run(self, noise_policy: NoisePolicy, rows: torch.Tensor, mode: str) -> dict:
        """One episode per env from grasp start ``rows``."""
        env, args, device = self.env, self.args, self.device
        n = env.num_envs
        source = SteeredDiffusionSource(self.diffusion, noise_policy, mode, bf16=args.bf16)
        controller = SuperMimicController(env, source, self.tracker)
        obs = self.start(rows)
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fallen, dropped, diverged = (torch.zeros(n, dtype=torch.bool, device=device) for _ in range(3))
        task = self.tracker_for()
        decisions_alive = []
        can = env.scene["can"]
        envs = torch.arange(n, device=device)
        k = source.execute_steps
        for step in range(round(args.episode_s / env.step_dt)):
            if step % k == 0:
                decisions_alive.append(alive.clone())
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            released = controller.plan_gripper[envs, controller.step_in_chunk - 1, 1] > self.task.release_opening
            # Images feeding the next replan: after control steps k-2 and k-1 of the chunk.
            self.render(step % k >= k - 2)
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(truncated.any()):
                raise RuntimeError("Env time-out inside a rollout")
            fell, drop, nan = self.failures(alive, terminated)
            fallen |= fell
            dropped |= drop
            diverged |= nan
            live = alive & ~terminated & ~nan
            task.step(can.data.root_link_pos_w, can.data.root_link_quat_w, released, live)
            # An episode ends at success, at a fall or with the can dropped.
            alive &= ~(task.placed | terminated | nan)
        self.render(True)
        returns = task.potential(fallen | dropped | diverged)
        records = source.records
        # Replans are synchronous (no resets inside a rollout): one record per decision.
        if any(int(r["rows"].sum()) != n for r in records):
            raise RuntimeError("Unexpected partial replan inside a rollout")
        cond = torch.stack([r["cond"] for r in records])  # (D, n, tokens, emb)
        noise = torch.stack([r["noise"] for r in records])  # (D, n, dim)
        log_prob = torch.stack([r["log_prob"] for r in records])  # (D, n)
        mask = torch.stack(decisions_alive[: len(records)])  # (D, n)
        # Masking multiplies by 0, which keeps NaN: drop non-finite decisions and zero their data.
        finite = cond.isfinite().flatten(2).all(-1) & noise.isfinite().all(-1) & log_prob.isfinite()
        mask &= finite
        return {
            "cond": torch.where(mask[..., None, None], cond, 0), "noise": torch.where(mask[..., None], noise, 0),
            "log_prob": torch.where(mask, log_prob, 0), "mask": mask.float(),
            "returns": returns, "lifted": task.lifted, "entered": task.entered, "placed": task.placed,
            "fallen": fallen, "dropped": dropped, "diverged": diverged, "final_dist": task.final_dist,
            "min_dist": task.min_dist,
        }  # fmt: skip
