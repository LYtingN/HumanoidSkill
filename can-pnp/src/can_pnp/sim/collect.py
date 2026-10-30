"""v3 simulated rollouts: object-WM data and the ranking check of the imagined chunk return.

:meth:`SimCollector.collect` rolls grasp-start episodes (8-step replans, as deployed) with a
fraction of the envs on N(0, I) noise, and records per 50 Hz step the right gripper pose, its
commanded opening, the can state and whether both fingers touch it; per decision the diffusion's
observation tokens, the robot state, the tracker observation and the env context (imagination
starts). Alongside it Phi runs a 32-step window from every decision on the goals the simulated
tracker actually received, so a window's imagined gripper poses and its simulated labels follow
the same references. Failed episodes are kept (their steps after the failure are masked).

:meth:`SimCollector.ranking` checks the imagined return against the simulator: every group of
envs starts from one grasp start and runs on shared noise (identical states) for a few
decisions, then each env executes its own candidate reference for the whole chunk; the
candidates' simulated chunk returns are compared with the imagined ones (Kendall tau per group).
All positions are in the env frame (env origin subtracted).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from can_pnp import constants as C
from can_pnp.body.imagine import PhiStep, unflatten_history
from can_pnp.body.kinematics import BodyKinematics
from can_pnp.dsrl import NoisePolicy, SteeredDiffusionSource
from can_pnp.layout import layout_term
from can_pnp.pipeline import SuperMimicController, gripper_opening
from can_pnp.sim.rollouts import Rollouts
from can_pnp.task import TaskTracker


@dataclass
class Episodes:
    """One collection rollout: T steps x N envs, D = T / 8 decisions, H-step Phi windows."""

    grip: torch.Tensor  # (T + 1, N, 7) right gripper base pose
    command: torch.Tensor  # (T, N) commanded right opening of each step
    obj: torch.Tensor  # (T + 1, N, 14) can pose / velocities, right finger opening
    held: torch.Tensor  # (T + 1, N) bool, both right fingers touch the can
    alive: torch.Tensor  # (T + 1, N) bool, the state belongs to the episode
    phi_grip: torch.Tensor  # (D, H + 1, N, 7) Phi's gripper poses of the window from decision d
    cond: torch.Tensor  # (D, N, tokens, emb) bf16 diffusion observation tokens
    s0: torch.Tensor  # (D, N, 71) robot state at the decision
    obs0: torch.Tensor  # (D, N, 930) tracker observation at the decision
    decision_alive: torch.Tensor  # (D, N) bool
    xi: torch.Tensor  # (N, 33)
    box: torch.Tensor  # (N, 3) box centre
    table: torch.Tensor  # (N, 3) table-top centre
    goal: torch.Tensor  # (N, 3) placement goal
    final_potential: torch.Tensor  # (N,) the episode's return (task potential at its end)
    base: torch.Tensor  # (N,) bool, the env ran on N(0, I) noise
    execute_steps: int = 8


def kendall_tau(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row of (G, K) scores: tau-a of the pairs not tied in ``b`` (NaN without such pairs)."""
    da = torch.sign(a[:, :, None] - a[:, None, :])
    db = torch.sign(b[:, :, None] - b[:, None, :])
    pairs = (db != 0).float().sum((1, 2))
    return (da * db).sum((1, 2)) / pairs


class SimCollector:
    def __init__(self, rollouts: Rollouts, phi: PhiStep, kin: BodyKinematics, horizon: int = 32):
        self.rollouts, self.phi, self.body, self.kin, self.horizon = rollouts, phi, phi.body, kin, horizon
        env = self.env = rollouts.env
        self.device = env.device
        robot = env.scene["robot"]
        self.robot, self.can = robot, env.scene["can"]
        self.grip_body = robot.body_names.index("right_gripper_base")
        self.torso = robot.body_names.index("torso_link")
        self.gripper_ids = robot.find_joints(C.GRIPPER_JOINT_NAMES, preserve_order=True)[0]
        self.fingers = [env.scene[f"finger_{s}_can_contacts"] for s in ("l", "r")]

    # ------------------------------------------------------------------ simulator state

    def grip_pose(self) -> torch.Tensor:
        data = self.robot.data
        pos = data.body_link_pos_w[:, self.grip_body] - self.env.scene.env_origins
        return torch.cat([pos, data.body_link_quat_w[:, self.grip_body]], dim=-1)

    def object_state(self) -> torch.Tensor:
        data = self.can.data
        opening = gripper_opening(self.robot.data.joint_pos[:, self.gripper_ids[1]])
        return torch.cat([data.root_link_pos_w - self.env.scene.env_origins, data.root_link_quat_w,
                          data.root_link_lin_vel_w, data.root_link_ang_vel_w, opening[:, None]], dim=-1)  # fmt: skip

    def held(self) -> torch.Tensor:
        found = [s.data.found.flatten(1).gt(0).any(-1) for s in self.fingers]
        return found[0] & found[1]

    def context(self) -> dict[str, torch.Tensor]:
        env, geo = self.env, self.rollouts.geo
        term = layout_term(env)
        origin = env.scene.env_origins
        table = torch.stack([
            term.layout["table_front_x"] + geo.table_size[0] / 2, term.layout["table_center_y"],
            torch.full_like(term.layout["table_center_y"], geo.table_height),
        ], dim=-1)  # fmt: skip
        return {"box": term.box_center_w - origin, "table": table, "goal": term.goal_pos_w - origin}

    def _goal(self, controller: SuperMimicController) -> torch.Tensor:
        """The goal the tracker got at the step just taken (``SuperMimicController.act``)."""
        return controller.plan.goal(controller.step_in_chunk - 1, self.robot.data.body_link_quat_w[:, self.torso])

    def _task(self) -> TaskTracker:
        ctx = self.context()
        zeros = torch.zeros(self.env.num_envs, device=self.device)
        return TaskTracker(self.rollouts.task, self.rollouts.geo, ctx["box"], ctx["goal"], zeros,
                           self.object_state()[:, :3], self.env.step_dt)  # fmt: skip

    # ------------------------------------------------------------------ collection

    @torch.no_grad()
    def collect(self, noise_policy: NoisePolicy, rows: torch.Tensor, base_envs: torch.Tensor) -> tuple[Episodes, dict]:
        """One episode per env from grasp start ``rows``; ``base_envs`` (N,) take N(0, I) noise."""
        ro, env, body, kin, device = self.rollouts, self.env, self.body, self.kin, self.device
        args, n, H = ro.args, self.env.num_envs, self.horizon
        source = SteeredDiffusionSource(ro.diffusion, noise_policy, "sample", bf16=args.bf16)
        source.base_envs = base_envs
        controller = SuperMimicController(env, source, ro.tracker)
        k = source.execute_steps
        steps = round(args.episode_s / env.step_dt)
        decisions = steps // k
        obs = ro.start(rows)
        xi = body.context()
        ctx = self.context()
        task = self._task()
        grip = torch.zeros(steps + 1, n, 7, device=device)
        obj = torch.zeros(steps + 1, n, 14, device=device)
        command = torch.zeros(steps, n, device=device)
        held = torch.zeros(steps + 1, n, dtype=torch.bool, device=device)
        alive_t = torch.zeros(steps + 1, n, dtype=torch.bool, device=device)
        phi_grip = torch.zeros(decisions, H + 1, n, 7, device=device)
        s0 = torch.zeros(decisions, n, 71, device=device)
        obs0 = torch.zeros(decisions, n, obs["policy"].shape[-1], device=device)
        decision_alive = torch.zeros(decisions, n, dtype=torch.bool, device=device)
        grip[0], obj[0], held[0], alive_t[0] = self.grip_pose(), self.object_state(), self.held(), True
        # Phi windows: one slot per decision of the last H steps.
        slots = -(-H // k)
        win_s = body.state().repeat(slots, 1, 1)
        win_hist = unflatten_history(obs["policy"]).unsqueeze(1).repeat(1, slots, 1, 1)  # (10, slots, N, 93)
        win_d = torch.full((slots,), -1, dtype=torch.long)
        xi_slots = xi.repeat(slots, 1)
        win_k = torch.zeros(slots, dtype=torch.long)
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fallen, dropped, diverged = (torch.zeros(n, dtype=torch.bool, device=device) for _ in range(3))
        envs = torch.arange(n, device=device)
        for step in range(steps):
            if step % k == 0 and step // k < decisions:
                d = step // k
                state = body.state()
                s0[d], obs0[d], decision_alive[d] = state, obs["policy"], alive
                slot = d % slots
                win_s[slot], win_hist[:, slot] = state, unflatten_history(obs["policy"])
                win_d[slot], win_k[slot] = d, 0
                phi_grip[d, 0] = grip[step]
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            command[step] = controller.plan_gripper[envs, controller.step_in_chunk - 1, 1]
            # Phi: every open window takes the step on the goal the simulated tracker just got.
            goal = self._goal(controller).repeat(slots, 1)
            _, s, _, hist = self.phi(win_hist.reshape(-1, slots * n, 93), goal, win_s.reshape(slots * n, 71), xi_slots)
            win_hist, win_s = hist.view(-1, slots, n, 93), s.view(slots, n, 71)
            poses = kin.poses(win_s)
            for slot in range(slots):
                if win_d[slot] >= 0 and win_k[slot] < H:
                    win_k[slot] += 1
                    phi_grip[win_d[slot], win_k[slot]] = torch.cat(
                        [poses["gripper_pos"][slot, :, 1], poses["gripper_quat"][slot, :, 1]], dim=-1
                    )
            released = command[step] > ro.task.release_opening
            ro.render(step % k >= k - 2)
            obs, _, terminated, _, _ = env.step(action)
            fell, drop, nan = ro.failures(alive, terminated)
            fallen |= fell
            dropped |= drop
            diverged |= nan
            live = alive & ~terminated & ~nan
            o = self.object_state()
            task.step(o[:, :3], o[:, 3:7], released, live)
            grip[step + 1], obj[step + 1], held[step + 1], alive_t[step + 1] = self.grip_pose(), o, self.held(), live
            alive &= ~(task.placed | terminated | nan)
        ro.render(True)
        records = source.records
        cond = torch.stack([r["cond"] for r in records[:decisions]])
        # Windows running past the episode end are incomplete: never sampled.
        complete = torch.arange(decisions, device=device) * k + H <= steps
        decision_alive &= complete[:, None]
        clean = torch.nan_to_num
        failed = fallen | dropped | diverged
        episodes = Episodes(
            final_potential=clean(task.potential(failed)), base=base_envs.clone(),
            grip=clean(grip), command=clean(command), obj=clean(obj), held=held, alive=alive_t,
            phi_grip=clean(phi_grip), cond=clean(cond), s0=clean(s0), obs0=clean(obs0),
            decision_alive=decision_alive, xi=xi, execute_steps=k, **ctx,
        )  # fmt: skip
        policy = ~base_envs
        stats = {}
        for name, m in (("sim", policy), ("sim_base", base_envs)):
            if bool(m.any()):
                stats.update({
                    f"{name}/return": float(task.potential(failed)[m].mean()),
                    f"{name}/success_rate": float(task.placed[m].float().mean()),
                    f"{name}/in_box_rate": float(task.entered[m].float().mean()),
                    f"{name}/drop_rate": float(dropped[m].float().mean()),
                    f"{name}/fall_rate": float(fallen[m].float().mean()),
                    f"{name}/final_dist": float(task.final_dist[m].mean()),
                    # Share of the episode steps with both right fingers on the can.
                    f"{name}/held_rate": float((held & alive_t)[:, m].sum() / alive_t[:, m].sum().clamp(min=1)),
                })  # fmt: skip
        return episodes, stats

    # ------------------------------------------------------------------ ranking check

    @torch.no_grad()
    def ranking(self, noise_policy: NoisePolicy, imaginer, rows: torch.Tensor, group: int, delay: int) -> dict:
        """Groups of ``group`` envs from grasp start ``rows`` (G,) share noise for ``delay``
        decisions, then try one candidate reference each for the whole chunk."""
        ro, env, body, device = self.rollouts, self.env, self.body, self.device
        n, H = env.num_envs, self.horizon
        if n % group:
            raise ValueError(f"{n} envs do not split into groups of {group}")
        source = SteeredDiffusionSource(ro.diffusion, noise_policy, "sample", bf16=ro.args.bf16)
        source.tie = group
        controller = SuperMimicController(env, source, ro.tracker)
        k = source.execute_steps
        obs = ro.start(rows.repeat_interleave(group))
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        for step in range(delay * k):
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            ro.render(step % k >= k - 2)
            obs, _, terminated, _, _ = env.step(action)
            alive &= ~terminated & ~ro.failures(alive, terminated)[2]
        # The decision: every env its own candidate, executed whole.
        state, obs_policy, obj0 = body.state(), obs["policy"].clone(), self.object_state()
        steps = round(ro.args.episode_s / env.step_dt)
        start = {"s0": state, "obs0": obs_policy, "xi": body.context(), "obj0": obj0, **self.context(),
                 "time": torch.full((n,), delay * k / steps, device=device), "step_frac": torch.full((n,), 1 / steps, device=device)}  # fmt: skip
        g = obj0[:, :3].view(-1, group, 3)
        grip = self.grip_pose()[:, :3].view(-1, group, 3)
        spread = torch.maximum((g - g[:, :1]).norm(dim=-1).amax(1), (grip - grip[:, :1]).norm(dim=-1).amax(1))
        source.tie = 1
        controller.execute_steps = H
        controller.step_in_chunk[:] = 0
        task = self._task()
        start_alive = alive.clone()
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        envs = torch.arange(n, device=device)
        ro.render(False)
        for step in range(H):
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            released = controller.plan_gripper[envs, controller.step_in_chunk - 1, 1] > ro.task.release_opening
            obs, _, terminated, _, _ = env.step(action)
            fell, drop, nan = ro.failures(alive, terminated)
            failed |= fell | drop | nan
            live = alive & ~terminated & ~nan
            o = self.object_state()
            task.step(o[:, :3], o[:, 3:7], released, live)
            alive &= ~(task.placed | terminated | nan)
        ro.render(True)
        record = source.records[-1]
        start["cond"] = record["cond"]
        d0 = (obj0[:, :3] - start["goal"]).norm(dim=-1)
        true = task.potential(failed) - torch.exp(-d0 / ro.task.dist_scale)
        imagined = imaginer.chunk_return(start, record["noise"])
        pred = imagined["returns"]
        tau = kendall_tau(pred.view(-1, group), true.view(-1, group))
        # Groups that broke up before the decision, or whose candidates all score the same, carry no ranking.
        ok = start_alive.view(-1, group).all(1) & tau.isfinite()
        # Final can position of the candidates whose episode did not end in a reset.
        kept = start_alive & ~failed
        err = (imagined["final_pos"] - self.object_state()[:, :3]).norm(dim=-1)
        return {
            "rank/tau": float(tau[ok].mean()) if bool(ok.any()) else float("nan"),
            "rank/groups": int(ok.sum()),
            "rank/return_mae": float((pred - true).abs()[start_alive].mean()),
            "rank/true_return_std": float(true.view(-1, group).std(1)[ok].mean()) if bool(ok.any()) else float("nan"),
            "rank/final_pos_err_cm": float(100 * err[kept].mean()) if bool(kept.any()) else float("nan"),
            "rank/clone_spread_cm": float(100 * spread.mean()),
        }  # fmt: skip
