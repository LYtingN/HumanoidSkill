"""v3 simulated rollouts: object-WM data and the ranking check of the imagined chunk return.

:meth:`SimCollector.collect` rolls grasp-start episodes (8-step replans, as deployed) with a
fraction of the envs on N(0, I) noise, and records per 50 Hz step the keypoints
(``KEYPOINT_BODIES``), both gripper openings and commands, the can state and whether both fingers
touch it; per decision the diffusion's observation tokens, the robot state, the tracker
observation and the env context (imagination starts). Alongside it Phi runs a 32-step window from
every decision on the goals the simulated tracker actually received, so a window's imagined
keypoints and its simulated labels follow the same references. Failed episodes are kept (their
steps after the failure are masked).

:meth:`SimCollector.ranking` checks the imagined return against the simulator: every group of
envs starts from one grasp start and runs on shared noise (identical states) for a few
decisions, then each env executes its own candidate reference for the whole chunk; the
candidates' simulated chunk returns are compared with the imagined ones (Kendall tau per group).
With a critic, every candidate's episode then continues on the noise policy to its end, and the
critic's prediction from the candidate's imagined window is compared with that episode return.
All positions are in the env frame (env origin subtracted).
"""

from __future__ import annotations

import torch

from hskill.data.episodes import Episodes
from hskill.embodiment import g1_aloha as C
from hskill.executors.imagine import PhiStep, reference_frames, unflatten_history
from hskill.executors.imaginer import ARM_JOINTS, ARM_STATE
from hskill.s0.controller import SuperMimicController, gripper_opening
from hskill.s0.kinematics import BodyKinematics
from hskill.s1.noise_policy import NoisePolicy, SteeredDiffusionSource
from hskill.sim.rollouts import Rollouts


def kendall_tau(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row of (G, K) scores: tau-a of the pairs not tied in ``b`` (NaN without such pairs)."""
    da = torch.sign(a[:, :, None] - a[:, None, :])
    db = torch.sign(b[:, :, None] - b[:, None, :])
    pairs = (db != 0).float().sum((1, 2))
    return (da * db).sum((1, 2)) / pairs


def tracking_stats(ep: Episodes, placed: torch.Tensor, failed: torch.Tensor, policy: torch.Tensor,
                   well: float) -> dict[str, float]:  # fmt: skip
    """How exactly S0 executed the references: right-hand error to the ideal branch's hand (the
    reference executed exactly; see :class:`SimCollector`). A window whose error stays below ``well``
    is an exact execution of its intent (its
    outcome is the ideal branch's ground truth); an episode that failed although tracked well
    failed on its intent, one that failed with large errors is a candidate execution failure."""
    err = (ep.ref_kp[..., C.RIGHT_EEF, :3] - ep.kp[..., C.RIGHT_EEF, :3]).norm(dim=-1)  # (T + 1, N)
    live = ep.alive
    k, H = ep.execute_steps, ep.phi_kp.shape[1] - 1
    t = torch.arange(ep.phi_kp.shape[0], device=err.device)[:, None] * k + torch.arange(1, H + 1, device=err.device)
    t = t.clamp(max=err.shape[0] - 1)  # incomplete windows are not in decision_alive anyway
    window = torch.where(live[t], err[t], 0).amax(1)  # (D, H, N) -> (D, N): worst step of each window
    win_ok = ep.decision_alive
    ep_err = torch.where(live, err, 0).amax(0)  # (N,) worst step of each episode
    lost = policy & ~placed
    stats = {
        "track/err_cm": 100 * float(err[live].mean()),
        "track/err_p95_cm": 100 * float(err[live].quantile(0.95)) if bool(live.any()) else float("nan"),
        "track/well_window_share": float((window[win_ok] < well).float().mean()) if bool(win_ok.any()) else float("nan"),
    }
    if bool(lost.any()):
        # Failures (not done) of the policy's episodes: arms tracked well throughout vs not.
        stats["track/fail_well_tracked_share"] = float((ep_err[lost] < well).float().mean())
        stats["track/fail_dropped_or_fell_share"] = float(failed[lost].float().mean())
    return stats


def _by_decision(records: list[dict | None], key: str, n: int) -> torch.Tensor:
    """(D, n, ...) of a field of the per-decision source records, which hold only the rows that
    replanned (``rows``); zeros elsewhere (envs whose episode was over: not decision_alive)."""
    ref = next(r[key] for r in records if r is not None)
    out = torch.zeros(len(records), n, *ref.shape[1:], dtype=ref.dtype, device=ref.device)
    for d, r in enumerate(records):
        if r is not None:
            out[d, r["rows"]] = r[key]
    return out


def _pair_check(true, pred, pred_ideal, start_alive, track_err, placed, size, ok) -> dict[str, float]:
    """Envs 2j (S0) and 2j + 1 (E*) executed the same candidate: the simulated J* (the E* env's chunk
    return) against the ideal branch's prediction, and the simulated gap J* - J against the imagined one."""
    s0, ideal = slice(0, None, 2), slice(1, None, 2)
    both = start_alive[s0] & start_alive[ideal]
    sim_gap, imag_gap = true[ideal] - true[s0], pred_ideal[s0] - pred[s0]

    def mean(x):
        return float(x[both].mean()) if bool(both.any()) else float("nan")

    corr = float("nan")
    if int(both.sum()) > 2:
        a, b = sim_gap[both] - sim_gap[both].mean(), imag_gap[both] - imag_gap[both].mean()
        corr = float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-9))
    tau = kendall_tau(pred_ideal[s0].view(-1, size), true[ideal].view(-1, size))
    keep = ok & tau.isfinite()
    return {
        "rank/sim_ideal_mae": mean((pred_ideal[s0] - true[ideal]).abs()), "rank/sim_ideal_return": mean(true[ideal]),
        "rank/sim_gap": mean(sim_gap), "rank/sim_gap_pos_share": mean((sim_gap > 0.05).float()),
        "rank/imag_gap": mean(imag_gap), "rank/gap_corr": corr,
        "rank/ideal_tau": float(tau[keep].mean()) if bool(keep.any()) else float("nan"),
        "rank/sim_ideal_placed": mean(placed[ideal].float()), "rank/sim_placed": mean(placed[s0].float()),
        "rank/ideal_exec_err_cm": 100 * mean(track_err[ideal]),
    }  # fmt: skip


class SimCollector:
    def __init__(self, rollouts: Rollouts, phi: PhiStep, kin: BodyKinematics, horizon: int = 32,
                 well_tracked: float = 0.02, history: int = 10, ideal: str = "full", phi_env_fraction: float = 1.0,
                 ideal_executor: bool = False, ideal_kp_scale: float = 10.0):  # fmt: skip
        if ideal not in ("full", "arms"):
            raise ValueError(f"ideal must be 'full' or 'arms', got {ideal!r}")
        self.rollouts, self.phi, self.body, self.kin, self.horizon = rollouts, phi, phi.body, kin, horizon
        # Ideal branch: the reference executed exactly ("full": the whole body, the root by the plan's
        # leg odometry from the replan state) or the actual body with the reference's arm joints ("arms").
        self.ideal = ideal
        self.history = history  # frames of keypoints / openings / object before an imagination start
        self.well_tracked = well_tracked  # m: right-hand tracking error of an "exactly executed" step
        # Phi's windows (the object WM's Phi-body inputs) on a random share of the envs only: they
        # cost about a fifth of a collection, and a share of the envs still gives thousands of windows.
        self.phi_env_fraction = phi_env_fraction
        env = self.env = rollouts.env
        self.device = env.device
        robot = env.scene["robot"]
        self.robot, self.can = robot, env.scene["can"]
        self.kp_bodies = [robot.body_names.index(b) for b in C.KEYPOINT_BODIES]
        self.torso = robot.body_names.index("torso_link")
        self.gripper_ids = robot.find_joints(C.GRIPPER_JOINT_NAMES, preserve_order=True)[0]
        self.fingers = [env.scene[f"finger_{s}_can_contacts"] for s in ("l", "r")]
        # The simulated ideal executor E* (hskill.executors.ideal_body: every body joint on a stiff
        # servo, the root held by a PD wrench), hooked into the env now and idle until a ranking check
        # pairs candidates (``ranking(ideal_pairs=True)``): the ideal branch's ground truth.
        self.executor = None
        if ideal_executor:
            from hskill.executors.ideal_body import ideal_executor as make

            self.executor = make(env, None, torch.zeros(env.num_envs, dtype=torch.bool, device=self.device), ideal_kp_scale)

    # ------------------------------------------------------------------ simulator state

    def keypoints(self) -> torch.Tensor:
        """(N, 5, 7) keypoint poses, env frame."""
        data = self.robot.data
        pos = data.body_link_pos_w[:, self.kp_bodies] - self.env.scene.env_origins[:, None]
        return torch.cat([pos, data.body_link_quat_w[:, self.kp_bodies]], dim=-1)

    def widths(self) -> torch.Tensor:
        """(N, 2) measured left / right gripper openings."""
        return gripper_opening(self.robot.data.joint_pos[:, self.gripper_ids])

    def object_state(self) -> torch.Tensor:
        data = self.can.data
        opening = gripper_opening(self.robot.data.joint_pos[:, self.gripper_ids[1]])
        return torch.cat([data.root_link_pos_w - self.env.scene.env_origins, data.root_link_quat_w,
                          data.root_link_lin_vel_w, data.root_link_ang_vel_w, opening[:, None]], dim=-1)  # fmt: skip

    def ideal_frames(self, controller: SuperMimicController, plan_start: torch.Tensor) -> torch.Tensor | None:
        """The current plan's frames as exactly executed body states (``ideal="full"``; else None),
        from the replan state ``plan_start`` (N, 71)."""
        return reference_frames(controller.plan, plan_start) if self.ideal == "full" else None

    def ideal_keypoints(self, controller: SuperMimicController, frames: torch.Tensor | None) -> torch.Tensor:
        """(N, 5, 7) the ideal branch's keypoints after the control step just taken (target frame
        ``step_in_chunk - 1`` of the plan): the exactly executed reference ``frames`` (F, N, 71), or
        the actual body with the plan's arm joints."""
        envs = torch.arange(self.env.num_envs, device=self.device)
        j = controller.step_in_chunk - 1
        if frames is not None:
            return self.kin.keypoints(frames[j, envs])
        state = self.body.state()
        state[:, ARM_STATE] = controller.plan.joint_pos[envs, j][:, ARM_JOINTS]
        return self.kin.keypoints(state)

    def held(self) -> torch.Tensor:
        found = [s.data.found.flatten(1).gt(0).any(-1) for s in self.fingers]
        return found[0] & found[1]

    def context(self) -> dict[str, torch.Tensor]:
        env, geo = self.env, self.rollouts.geo
        term = self.rollouts.layout_term(env)
        origin = env.scene.env_origins
        table = torch.stack([
            term.layout["table_front_x"] + geo.table_size[0] / 2, term.layout["table_center_y"],
            torch.full_like(term.layout["table_center_y"], geo.table_height),
        ], dim=-1)  # fmt: skip
        return {"box": term.box_center_w - origin, "table": table, "goal": term.goal_pos_w - origin}

    def _goal(self, controller: SuperMimicController) -> torch.Tensor:
        """The goal the tracker got at the step just taken (``SuperMimicController.act``)."""
        return controller.plan.goal(controller.step_in_chunk - 1, self.robot.data.body_link_quat_w[:, self.torso])

    def _task(self):
        ctx = self.context()
        zeros = torch.zeros(self.env.num_envs, device=self.device)
        return self.rollouts.task_tracker(self.rollouts.task, self.rollouts.geo, ctx["box"], ctx["goal"], zeros,
                           self.object_state()[:, :3], self.env.step_dt)  # fmt: skip

    # ------------------------------------------------------------------ collection

    @torch.no_grad()
    def collect(self, noise_policy: NoisePolicy, rows: torch.Tensor, base_envs: torch.Tensor,
                episode_s: float | None = None) -> tuple[Episodes, dict]:  # fmt: skip
        """One episode per env from grasp start ``rows``; ``base_envs`` (N,) take N(0, I) noise.
        ``episode_s`` overrides ``args.episode_s`` (e.g. per curriculum phase)."""
        ro, env, body, kin, device = self.rollouts, self.env, self.body, self.kin, self.device
        args, n, H = ro.args, self.env.num_envs, self.horizon
        source = SteeredDiffusionSource(ro.diffusion, noise_policy, "sample", bf16=args.bf16)
        source.base_envs = base_envs
        controller = SuperMimicController(env, source, ro.tracker)
        k = source.execute_steps
        steps = round((episode_s or args.episode_s) / env.step_dt)
        decisions = steps // k
        obs = ro.start(rows)
        xi = body.context()
        ctx = self.context()
        task = self._task()
        kp = torch.zeros(steps + 1, n, 5, 7, device=device)
        width = torch.zeros(steps + 1, n, 2, device=device)
        obj = torch.zeros(steps + 1, n, 14, device=device)
        command = torch.zeros(steps, n, 2, device=device)
        held = torch.zeros(steps + 1, n, dtype=torch.bool, device=device)
        ref_kp = torch.zeros(steps + 1, n, 5, 7, device=device)
        alive_t = torch.zeros(steps + 1, n, dtype=torch.bool, device=device)
        phi_kp = torch.zeros(decisions, H + 1, n, 5, 7, device=device)
        s0 = torch.zeros(decisions, n, 71, device=device)
        obs0 = torch.zeros(decisions, n, obs["policy"].shape[-1], device=device)
        decision_alive = torch.zeros(decisions, n, dtype=torch.bool, device=device)
        kp[0], width[0], obj[0], held[0], alive_t[0] = self.keypoints(), self.widths(), self.object_state(), self.held(), True
        ref_kp[0] = kp[0]
        # Phi windows: one slot per decision of the last H steps.
        slots = -(-H // k)
        pe = torch.randperm(n, device=device)[: max(1, round(self.phi_env_fraction * n))].sort().values  # Phi's envs
        m = len(pe)
        win_s = body.state()[pe].repeat(slots, 1, 1)
        win_hist = unflatten_history(obs["policy"])[:, pe].unsqueeze(1).repeat(1, slots, 1, 1)  # (10, slots, m, 93)
        win_d = torch.full((slots,), -1, dtype=torch.long)
        xi_slots = xi[pe].repeat(slots, 1)
        win_k = torch.zeros(slots, dtype=torch.long)
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fallen, dropped, diverged = (torch.zeros(n, dtype=torch.bool, device=device) for _ in range(3))
        envs = torch.arange(n, device=device)
        decision_records: list[dict | None] = [None] * decisions  # the source's record of each decision
        for step in range(steps):
            # Envs whose episode is over keep replaying their last plan: no encoding, no sampling.
            controller.frozen = ~alive
            if step % k == 0 and step // k < decisions:
                d = step // k
                state = body.state()
                s0[d], obs0[d], decision_alive[d] = state, obs["policy"], alive
                slot = d % slots
                win_s[slot], win_hist[:, slot] = state[pe], unflatten_history(obs["policy"])[:, pe]
                win_d[slot], win_k[slot] = d, 0
                phi_kp[d, 0] = kp[step]
            records_before = len(source.records)
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            if step % k == 0 and step // k < decisions and len(source.records) > records_before:
                decision_records[step // k] = source.records[-1]
            if step % k == 0:  # every env replanned (no resets inside a rollout; ended ones replay)
                frames = self.ideal_frames(controller, s0[step // k] if step // k < decisions else body.state())
            command[step] = controller.plan_gripper[envs, controller.step_in_chunk - 1]
            # Phi: every open window takes the step on the goal the simulated tracker just got.
            goal = self._goal(controller)[pe].repeat(slots, 1)
            _, s, _, hist = self.phi(win_hist.reshape(-1, slots * m, 93), goal, win_s.reshape(slots * m, 71), xi_slots)
            win_hist, win_s = hist.view(-1, slots, m, 93), s.view(slots, m, 71)
            win_kp = kin.keypoints(win_s)
            for slot in range(slots):
                if win_d[slot] >= 0 and win_k[slot] < H:
                    win_k[slot] += 1
                    phi_kp[win_d[slot], win_k[slot], pe] = win_kp[slot]
            released = command[step, :, 1] > ro.task.release_opening
            ro.render(step % k >= k - 2)
            obs, _, terminated, _, _ = env.step(action)
            fell, drop, nan = ro.failures(alive, terminated)
            fallen |= fell
            dropped |= drop
            diverged |= nan
            live = alive & ~terminated & ~nan
            o, holds = self.object_state(), self.held()
            task.step(o[:, :3], o[:, 3:7], released, live, holds)
            dropped |= live & task.toppled  # a toppled can ends the episode like a dropped one
            kp[step + 1], width[step + 1], obj[step + 1] = self.keypoints(), self.widths(), o
            held[step + 1], alive_t[step + 1] = holds, live
            ref_kp[step + 1] = self.ideal_keypoints(controller, frames)
            alive &= ~(task.placed | task.toppled | terminated | nan)
        ro.render(True)
        cond, noise, log_prob = (_by_decision(decision_records, key, n) for key in ("cond", "noise", "log_prob"))
        decision_live = decision_alive.clone()
        # Windows running past the episode end are incomplete: never sampled.
        complete = torch.arange(decisions, device=device) * k + H <= steps
        decision_alive &= complete[:, None]
        clean = torch.nan_to_num
        failed = fallen | dropped | diverged
        episodes = Episodes(
            final_potential=clean(task.potential(failed)), base=base_envs.clone(),
            kp=clean(kp), width=clean(width), command=clean(command), obj=clean(obj), held=held, alive=alive_t,
            phi_kp=clean(phi_kp), cond=clean(cond), s0=clean(s0), obs0=clean(obs0),
            phi_env=torch.zeros(n, dtype=torch.bool, device=device).index_fill_(0, pe, True),
            decision_alive=decision_alive, xi=xi, execute_steps=k, ref_kp=clean(ref_kp),
            noise=clean(noise), log_prob=clean(log_prob), decision_live=decision_live,
            end_step=alive_t.sum(0) - 1, placed=task.placed.clone(), dropped=dropped & ~task.placed,
            fallen=(fallen | diverged) & ~task.placed, **ctx,
        )  # fmt: skip
        policy = ~base_envs
        stats = tracking_stats(episodes, task.placed, failed, policy, self.well_tracked)
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
    def ranking(self, noise_policy: NoisePolicy, imaginer, rows: torch.Tensor, group: int, delay: int,
                critic=None, episode_s: float | None = None, uncertainty_max: float | None = None,
                ideal_pairs: bool = False) -> dict:  # fmt: skip
        """Groups of ``group`` envs from grasp start ``rows`` (G,) share noise for ``delay``
        decisions, then try one candidate reference each for one executed chunk (the steps up to the
        next replan, as collected; the imagination scores the same steps), and with a ``critic``
        continue on the noise policy to the episode's end.

        ``ideal_pairs``: envs 2j and 2j + 1 try the same candidate, 2j with S0 and 2j + 1 with the
        simulated ideal executor E* (back to S0 after the chunk): each candidate's simulated J*
        next to its J, for the chunk and (with a ``critic``) for the episode. The ranking statistics
        then use the S0 envs only (``group // 2`` candidates per group)."""
        ro, env, body, device = self.rollouts, self.env, self.body, self.device
        n = env.num_envs
        if n % group or (ideal_pairs and group % 2):
            raise ValueError(f"{n} envs do not split into groups of {group}" + (" pairs" if ideal_pairs else ""))
        if ideal_pairs and self.executor is None:
            raise ValueError("ideal_pairs needs the collector's ideal executor (ideal_executor=True)")
        source = SteeredDiffusionSource(ro.diffusion, noise_policy, "sample", bf16=ro.args.bf16)
        source.tie = group
        controller = SuperMimicController(env, source, ro.tracker)
        k = source.execute_steps
        if self.executor is not None:
            for e in self.executor:
                e.attach(controller)
        obs = ro.start(rows.repeat_interleave(group))
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        frames = [(self.keypoints(), self.widths(), self.object_state()[:, :10])]  # history of the start
        for step in range(delay * k):
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            ro.render(step % k >= k - 2)
            obs, _, terminated, _, _ = env.step(action)
            alive &= ~terminated & ~ro.failures(alive, terminated)[2]
            frames = (frames + [(self.keypoints(), self.widths(), self.object_state()[:, :10])])[-self.history :]
        # The decision: every env its own candidate, executed whole.
        state, obs_policy, obj0 = body.state(), obs["policy"].clone(), self.object_state()
        steps = round((episode_s or ro.args.episode_s) / env.step_dt)
        frames = [frames[0]] * (self.history - len(frames)) + frames  # before the episode: its first frame
        start = {"s0": state, "obs0": obs_policy, "xi": body.context(), "obj0": obj0, **self.context(),
                 "hist_kp": torch.stack([f[0] for f in frames], 1), "hist_width": torch.stack([f[1] for f in frames], 1),
                 "hist_obj": torch.stack([f[2] for f in frames], 1),
                 "time": torch.full((n,), delay * k / steps, device=device), "step_frac": torch.full((n,), 1 / steps, device=device)}  # fmt: skip
        g = obj0[:, :3].view(-1, group, 3)
        grip = self.keypoints()[:, C.RIGHT_EEF, :3].view(-1, group, 3)
        spread = torch.maximum((g - g[:, :1]).norm(dim=-1).amax(1), (grip - grip[:, :1]).norm(dim=-1).amax(1))
        source.tie = 2 if ideal_pairs else 1
        controller.step_in_chunk[:] = 0
        envs = torch.arange(n, device=device)
        on_ideal = (envs % 2 == 1) if ideal_pairs else torch.zeros(n, dtype=torch.bool, device=device)
        if ideal_pairs:
            for e in self.executor:
                e.set_active(on_ideal.nonzero().flatten(), True)
        task = self._task()
        phi0 = task.shaping()  # the task's shaping at the decision
        start_alive = alive.clone()
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        track_err = torch.zeros(n, device=device)
        ro.render(False)
        for step in range(k):
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            if step == 0:  # the candidates' plans, made from the decision state
                frames = self.ideal_frames(controller, state)
            released = controller.plan_gripper[envs, controller.step_in_chunk - 1, 1] > ro.task.release_opening
            if critic is not None:  # the images of the replan after the chunk
                ro.render(step >= k - 2)
            obs, _, terminated, _, _ = env.step(action)
            ideal = self.ideal_keypoints(controller, frames)
            hand_err = (ideal[:, C.RIGHT_EEF, :3] - self.keypoints()[:, C.RIGHT_EEF, :3]).norm(dim=-1)
            track_err = torch.where(alive, torch.maximum(track_err, hand_err), track_err)
            fell, drop, nan = ro.failures(alive, terminated)
            failed |= fell | drop | nan
            live = alive & ~terminated & ~nan
            o = self.object_state()
            task.step(o[:, :3], o[:, 3:7], released, live, self.held())
            failed |= live & task.toppled
            alive &= ~(task.placed | task.toppled | terminated | nan)
        ro.render(True)
        if ideal_pairs:  # back to S0 for the rest of the episode
            for e in self.executor:
                e.set_active(on_ideal.nonzero().flatten(), False)
        record = source.records[-1]
        start["cond"] = record["cond"]
        true = task.potential(failed) - phi0
        imagined = imaginer.chunk_return(start, record["noise"], k)
        # The simulator only sees the chunk: compare the imagined chunk outcomes (object WM on Phi's
        # hands), not the critic's estimate of the episode beyond it. Ranked: the S0 envs.
        s0 = ~on_ideal
        size = group // 2 if ideal_pairs else group  # candidates per group
        pred = imagined["chunk_returns"]
        tau = kendall_tau(pred[s0].view(-1, size), true[s0].view(-1, size))
        # Groups that broke up before the decision, or whose candidates all score the same, carry no ranking.
        ok = start_alive.view(-1, group).all(1) & tau.isfinite()
        # Final can position of the candidates whose episode did not end in a reset.
        kept = start_alive & ~failed & s0
        err = (imagined["final_pos"] - self.object_state()[:, :3]).norm(dim=-1)
        # Ideal branch: where S0 executed the candidate's arms exactly, the simulated return is the
        # intent's own value, which the ideal branch predicts; elsewhere the predicted gap should be larger.
        # Only candidates the object WM's members agree on for the ideal branch (an input it rarely sees).
        trusted = s0 & (start_alive if uncertainty_max is None else start_alive & (imagined["uncertainty_ideal"] <= uncertainty_max))
        exact = trusted & (track_err < self.well_tracked)
        gap = imagined["chunk_returns_ideal"] - pred
        ideal_check = {
            "rank/ideal_trusted_share": float(trusted.float().sum() / (start_alive & s0).float().sum().clamp(min=1)),
            "rank/exact_share": float(exact.float().sum() / s0.float().sum()),
            "rank/ideal_mae_exact": float((imagined["chunk_returns_ideal"] - true).abs()[exact].mean()) if bool(exact.any()) else float("nan"),
            "rank/gap_exact": float(gap[exact].mean()) if bool(exact.any()) else float("nan"),
            "rank/gap_inexact": float(gap[trusted & ~exact].mean()) if bool((trusted & ~exact).any()) else float("nan"),
            "rank/track_err_cm": float(100 * track_err[start_alive & s0].mean()) if bool((start_alive & s0).any()) else float("nan"),
        }  # fmt: skip
        if ideal_pairs:
            ideal_check |= _pair_check(true, pred, imagined["chunk_returns_ideal"], start_alive, track_err, task.placed, size, ok)
        critic_check = {} if critic is None else self._critic_check(
            critic, controller, source, task, obs, reset, alive, failed, start_alive, imagined, pred,
            start["goal"], (steps - delay * k) * env.step_dt, delay * k + k, steps, group, on_ideal,
        )  # fmt: skip
        if self.executor is not None:
            for e in self.executor:
                e.attach(None)
        return {
            **ideal_check, **critic_check,
            "rank/tau": float(tau[ok].mean()) if bool(ok.any()) else float("nan"),
            "rank/groups": int(ok.sum()),
            "rank/return_mae": float((pred - true).abs()[start_alive & s0].mean()),
            "rank/true_return_std": float(true[s0].view(-1, size).std(1)[ok].mean()) if bool(ok.any()) else float("nan"),
            "rank/final_pos_err_cm": float(100 * err[kept].mean()) if bool(kept.any()) else float("nan"),
            "rank/clone_spread_cm": float(100 * spread.mean()),
        }  # fmt: skip

    def _critic_check(self, critic, controller, source, task, obs, reset, alive, failed, start_alive, imagined,
                      chunk_pred, goal, t_left: float, first: int, steps: int, group: int,
                      on_ideal: torch.Tensor) -> dict:  # fmt: skip
        """``t_left``: seconds of the episode left at the decision."""
        """Continue the candidates' episodes from the chunk's end on the noise policy (8-step replans,
        as collected) and compare their returns with the critic's prediction from the candidates'
        imagined windows, and with the imagined chunk return alone."""
        from hskill.learners.critic import critic_features

        ro, env = self.rollouts, self.env
        envs = torch.arange(env.num_envs, device=self.device)
        source.tie = 1
        controller.execute_steps = source.execute_steps  # the chunk's step count is a multiple: replan now
        k = source.execute_steps
        for j in range(steps - first):
            action = controller.act(obs, reset)
            released = controller.plan_gripper[envs, controller.step_in_chunk - 1, 1] > ro.task.release_opening
            ro.render(j % k >= k - 2)
            obs, _, terminated, _, _ = env.step(action)
            fell, drop, nan = ro.failures(alive, terminated)
            failed |= fell | drop | nan
            live = alive & ~terminated & ~nan
            o = self.object_state()
            task.step(o[:, :3], o[:, 3:7], released, live, self.held())
            failed |= live & task.toppled
            alive &= ~(task.placed | task.toppled | terminated | nan)
        ro.render(True)
        true = task.potential(failed)
        n = env.num_envs
        feat = critic_features(self.body, imagined["s_end"], imagined["kp_end"], imagined["obj_end"],
                               imagined["held_logit_end"], goal, torch.full((n,), t_left, device=self.device))  # fmt: skip
        members = critic(feat)
        q = members.mean(0)
        s0 = ~on_ideal  # ranked: the S0 envs (with E* pairs, the other half executed the chunk ideally)
        size = int(s0.sum()) * group // len(s0)

        def tau_of(x):
            return kendall_tau(x[s0].view(-1, size), true[s0].view(-1, size))

        tau_q, tau_chunk = tau_of(q), tau_of(chunk_pred)
        # The imagined score as the imagined updates use it (chunk outcome, or the critic beyond it).
        tau_score = tau_of(imagined["returns"])
        ok = start_alive.view(-1, group).all(1) & tau_q.isfinite()
        m = start_alive & s0

        def mean(x, mask):
            return float(x[mask].mean()) if bool(mask.any()) else float("nan")

        pairs = {}
        if bool(on_ideal.any()):  # the episode's return with one decision executed ideally, minus with S0
            both = start_alive[s0] & start_alive[on_ideal]
            gap = true[on_ideal] - true[s0]
            pairs = {"rank/episode_gap": mean(gap, both), "rank/episode_gap_pos_share": mean((gap > 0.05).float(), both),
                     "rank/episode_return_ideal": mean(true[on_ideal], both)}  # fmt: skip
        return {
            **pairs,
            "rank/critic_tau": mean(tau_q, ok), "rank/chunk_tau_episode": mean(tau_chunk, ok),
            "rank/score_tau_episode": mean(tau_score, ok),
            "rank/critic_groups": int(ok.sum()), "rank/critic_mae": mean((q - true).abs(), m),
            "rank/episode_return": mean(true, m), "rank/episode_success": mean(task.placed.float(), m),
            "rank/episode_return_group_std": mean(true[s0].view(-1, size).std(1), ok),
            "rank/critic_group_std": mean(q[s0].view(-1, size).std(1), ok),
            "rank/critic_member_std": mean(members.std(0), m),
        }  # fmt: skip
