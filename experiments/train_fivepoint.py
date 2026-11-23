"""Fine-tune five-point FM latent actions with FlashSAC and a frozen v0.1 human/five tracker.

Run experiments/launch/train_fivepoint_flashsac.sh --help from the checkout.
One process per GPU under torchrun, each simulating --num-envs lockstep worlds with batched FM
and tracker inference. Every rank stores its own replay shard and runs the same FlashSAC update on
its local batch; gradients and batch-norm statistics are averaged. A step (iteration) is one FM
decision in every world of every rank.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

from hskill.learners.flashsac.learner import FlashSAC, Replay


# Per-episode outcomes summed over a log window (means are reported).
EPISODE_STATS = ("return", "length_s", "success", "held", "lifted", "entered_box", "toppled", "thrown",
                 "env_terminated", "min_gripper_can_m", "min_can_goal_m")
PHASE_STATS = ("success", "held", "lifted", "entered_box", "toppled", "thrown", "env_terminated", "return")


TRACKER_ROWS = ("box", "goal", "origin_z", "lifted", "entered", "placed", "toppled", "hold", "tilted",
                "thrown", "flung", "final_dist", "min_dist", "last_pos")


PHASES = ("APPROACH", "REACH", "CLOSE", "LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE")


class FivePointTask:
    """N lockstep worlds. A world whose episode ends inside a chunk idles to the chunk's end and is
    reset at the next decision; its remaining control steps are not accounted.

    Reset bank (SGS, arXiv:2610.12465): every (phase, training episode) start of the recorded
    collection, fixed before training. Each world draws its next configuration i.i.d. by SGS weight;
    its time budget is the demonstration's time from that start to RELEASE, times ``slack``, plus
    ``margin`` seconds, capped at ``seconds``."""

    def __init__(self, env, controller, geometry, seconds, gamma, seed, sampler_seed,
                 phases=PHASES, slack=1.5, margin=4.0, eval_envs=0, eval_offset=0, measured_gripper=True,
                 reach_weight=0.0, dense_starts=False, task_cfg=None, noise_scale=3.0, noise_repeat=False):
        from can_pnp.starts import DenseStarts, GraspStarts, VAL_EPISODES
        from hskill.data.sgs import SuccessGuidedSampler
        from hskill.embodiment.g1_aloha import KEYPOINT_BODIES, RIGHT_EEF
        from supermimic.world_model.state_spec import StateSpec, StateSpecCfg

        from can_pnp.task_spec import TaskCfg

        self.env, self.ctrl, self.geometry = env, controller, geometry
        self.task_cfg = task_cfg or TaskCfg()
        # FM latent = noise_scale x bounded SAC action: one frame shared by the whole chunk when repeated
        # (DSRL's pi0 parameterization), else one latent per chunk frame.
        self.noise_scale, self.noise_repeat = noise_scale, noise_repeat
        self.n, device = env.num_envs, env.device
        self.gamma = gamma
        if "RELEASE" not in phases:
            raise ValueError("Time budgets are measured to the RELEASE phase; include it")
        # Dense starts: every 0.5 s of each episode (measured gripper openings), labelled by phase.
        self.starts = (DenseStarts(phase=tuple(phases)) if dense_starts
                       else GraspStarts(phase=tuple(phases), measured_gripper=measured_gripper))
        phase_start = getattr(self.starts, "phase_start", torch.ones(len(self.starts), dtype=torch.bool))
        rows = [i for i, ep in enumerate(self.starts.episode_ids) if ep not in VAL_EPISODES]
        self.sampler = SuccessGuidedSampler(self.starts, rows, sampler_seed)
        # Per configuration: demonstration seconds from its start to the same episode's RELEASE.
        release = {ep: f for ep, f, ph, first in zip(self.starts.episode_ids, self.starts.frame.tolist(),
                                                      self.starts.phase_ids.tolist(), phase_start.tolist())
                   if phases[ph] == "RELEASE" and first}
        demo_s = torch.tensor([(release[ep] - f) / 50.0 for ep, f in zip(self.starts.episode_ids,
                                                                        self.starts.frame.tolist())])
        budget_s = (slack * demo_s.clamp_min(0) + margin).clamp_max(seconds)
        self.budget = (budget_s / env.step_dt).round().long().clamp_min(1).to(device)
        self.phase_names = tuple(phases)
        # Reach shaping (weight 0: off). Before the can is first held, a potential of the right gripper's
        # distance to the can beyond the held distance; once held it stays at its maximum, so grasping
        # never lowers the potential. Starts recorded with the can in hand begin at the maximum.
        self.reach_weight = reach_weight
        self.held_start = torch.tensor([p in HELD_PHASES for p in self.phase_names], device=env.device)
        print("Episode budget per phase (s, median): " + ", ".join(
            f"{p} {budget_s[self.starts.phase_ids == i].median().item():.1f}" for i, p in enumerate(phases)), flush=True)
        self.limit = torch.ones(self.n, dtype=torch.long, device=device)
        self.phase = torch.zeros(self.n, dtype=torch.long, device=device)
        # The last eval_envs worlds evaluate the deterministic policy from the normal (APPROACH)
        # start of every recorded episode in turn, validation episodes included. They feed no
        # replay, SGS or reward statistics.
        if not 0 <= eval_envs < self.n:
            raise ValueError("eval_envs must leave at least one training world")
        if eval_envs and "APPROACH" not in phases:
            raise ValueError("Evaluation starts from the APPROACH phase; include it")
        self.eval_mask = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.eval_mask[self.n - eval_envs:] = True
        approach = phases.index("APPROACH") if "APPROACH" in phases else -1
        self.eval_rows = torch.tensor([i for i, (ph, first) in enumerate(zip(self.starts.phase_ids.tolist(), phase_start.tolist()))
                                       if ph == approach and first])
        self.eval_val = torch.tensor([ep in VAL_EPISODES for ep in self.starts.episode_ids])
        self.eval_cursor = eval_offset
        self.generator = torch.Generator().manual_seed(seed)
        self.can = env.scene["can"].data
        robot = controller.robot
        self.kp = [robot.body_names.index(name) for name in KEYPOINT_BODIES]
        self.right_eef = self.kp[RIGHT_EEF]
        idx = robot.indexing
        self.q = torch.cat([idx.free_joint_q_adr, idx.joint_q_adr[controller.body_ids]])
        self.v = torch.cat([idx.free_joint_v_adr, idx.joint_v_adr[controller.body_ids]])
        self.spec = StateSpec(StateSpecCfg(), device)
        self.fingers = [env.scene[f"finger_{side}_can_contacts"] for side in ("l", "r")]
        self.row = torch.zeros(self.n, dtype=torch.long)
        self.policy_episode = torch.ones(self.n, dtype=torch.bool, device=device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.episode_return = torch.zeros(self.n, device=device)
        self.held_any = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.grasped = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.min_grip = torch.full((self.n,), math.inf, device=device)
        self.task = None
        self.finished = None  # (stats (n, len(EPISODE_STATS)), done mask) of the last step
        self.nonfinite_worlds = 0  # episodes ended by a nonfinite command

    def held(self):
        left, right = [finger.data.found.flatten(1).gt(0).any(-1) for finger in self.fingers]
        return left & right

    def state(self, ids=None):
        from hskill.learners.critic import task_state_features

        env, data = self.env, self.ctrl.robot.data
        ids = slice(None) if ids is None else ids
        origin = env.scene.env_origins[ids]
        s = torch.cat([env.sim.data.qpos[ids][:, self.q], env.sim.data.qvel[ids][:, self.v]], -1)
        s[:, :3] -= origin
        kp = torch.cat((data.body_link_pos_w[ids][:, self.kp] - origin[:, None], data.body_link_quat_w[ids][:, self.kp]), -1)
        can = torch.cat((self.can.root_link_pos_w[ids] - origin, self.can.root_link_quat_w[ids],
                         self.can.root_link_lin_vel_w[ids]), -1)
        left = (self.limit[ids] - self.steps[ids]).float() * env.step_dt
        return task_state_features(self.spec.canonical(s), s, kp, can, self.held()[ids].float(),
                                   self.task.goal[ids] - origin, left)

    def observation(self, ids=None):
        return self.ctrl.encode(ids).flatten(1), self.state(ids)

    @torch.no_grad()
    def reset(self, mask=None):
        """Start new episodes in the worlds of ``mask`` (all when None); their observations."""
        from can_pnp.scene.layout import layout_term
        from can_pnp.starts import apply_grasp_start
        from can_pnp.task_spec import TaskTracker

        device = self.env.device
        ids = (torch.arange(self.n, device=device) if mask is None else mask.nonzero().flatten())
        self.env.reset(env_ids=ids)
        rows = self.sampler.sample(len(ids), self.generator)
        evaluating = self.eval_mask[ids].cpu()
        if evaluating.any():
            k = int(evaluating.sum())
            picks = (self.eval_cursor + torch.arange(k)) % len(self.eval_rows)
            self.eval_cursor += k
            rows[evaluating] = self.eval_rows[picks]
        self.row[ids.cpu()] = rows
        self.limit[ids] = self.budget[rows.to(device)]
        self.phase[ids] = self.starts.phase_ids[rows].to(device)
        apply_grasp_start(self.env, ids, self.starts, rows)
        layout = layout_term(self.env)
        fresh = TaskTracker(self.task_cfg, self.geometry, layout.box_center_w.clone(), layout.goal_pos_w.clone(),
                            self.env.scene.env_origins[:, 2], self.can.root_link_pos_w, self.env.step_dt)
        if self.task is None:
            self.task = fresh
        else:
            for name in TRACKER_ROWS:
                getattr(self.task, name)[ids] = getattr(fresh, name)[ids]
        self.policy_episode[ids] = True
        self.steps[ids] = 0
        self.episode_return[ids] = 0
        self.held_any[ids] = False
        self.grasped[ids] = self.held_start[self.phase[ids]]
        self.min_grip[ids] = math.inf
        self.ctrl.reset_worlds(ids)
        return self.observation(ids)

    def gripper_can_distance(self):
        return (self.ctrl.robot.data.body_link_pos_w[:, self.right_eef] - self.can.root_link_pos_w).norm(dim=-1)

    def reach_potential(self):
        """In [0, 1]: 1 once the can has been held, else of the gripper's distance beyond the held distance."""
        excess = (self.gripper_can_distance() - HELD_DISTANCE).clamp_min(0.)
        phi = sum(w * torch.exp(-excess / s) for s, w in zip(REACH_SCALES, (.5, .5)))
        return torch.where(self.grasped, 1., phi)

    def potential(self):
        """Shaping potential: the can's distance to the goal, plus the weighted reach potential."""
        phi = self.task.shaping()
        return phi + self.reach_weight * self.reach_potential() if self.reach_weight else phi

    @torch.no_grad()
    def step(self, obs, state, action, *, policy_controlled):
        if not policy_controlled:
            self.policy_episode.zero_()
        ctrl, fm, task, n = self.ctrl, self.ctrl.fm, self.task, self.n
        cond = obs.reshape(n, fm.n_tokens, fm.n_emb)
        # Q and replay use the bounded SAC action. Only the FM receives the scaled latent.
        frames = 1 if self.noise_repeat else fm.horizon
        noise = (self.noise_scale * action.reshape(n, frames, fm.action_dim)).expand(n, fm.horizon, fm.action_dim)
        ctrl.set_plan(ctrl.sample(cond, noise))
        before = self.potential()
        live = torch.ones(n, dtype=torch.bool, device=obs.device)
        terminal = torch.zeros_like(live)
        auto_any = torch.zeros_like(live)
        for k in range(fm.execute_steps):
            # Render only the frames the next decision's observation history keeps.
            ctrl.render = k >= fm.execute_steps - fm.obs_horizon
            command = ctrl.execute_plan()
            released = ctrl.commanded_width[:, 1] > task.cfg.release_opening
            _, _, terminated, truncated, _ = self.env.step(command)
            # A nonfinite command ends the episode like an environment termination (no bootstrap).
            broken = ctrl.nonfinite & live
            if broken.any():
                self.nonfinite_worlds += int(broken.sum())
                print(f"{int(broken.sum())} world(s) had a nonfinite command; their episodes end", flush=True)
            auto = (terminated | truncated | ctrl.nonfinite) & live
            self.steps += live.long()
            held = self.held()
            accounted = live & ~auto
            task.step(self.can.root_link_pos_w, self.can.root_link_quat_w, released, accounted, held)
            self.held_any |= held & accounted
            self.grasped |= held & accounted
            self.min_grip = torch.where(accounted, torch.minimum(self.min_grip, self.gripper_can_distance()),
                                        self.min_grip)
            ends = live & (auto | task.placed | task.toppled | task.thrown)
            terminal |= ends
            auto_any |= auto
            live &= ~(ends | (self.steps >= self.limit))
            if ctrl.render:
                ctrl.push_history()
            if not live.any():
                break  # every world restarts, which refreshes its history
        ctrl.render = True
        done = ~live
        # Finished worlds have no bootstrap state. Mark them absorbing, including worlds
        # that finished early and idled while the rest of the chunk executed.
        next_obs, next_state = self.observation()
        next_obs.masked_fill_(done[:, None], 0.)
        next_state.masked_fill_(done[:, None], 0.)
        phi = torch.where(terminal, 0., self.potential())
        reward = task.terminal_reward() + self.gamma * phi - before
        self.sampler.observe(self.row, task.placed, done & ~self.eval_mask & self.policy_episode)
        self.episode_return += reward
        stats = torch.stack([self.episode_return, self.steps.float() * self.env.step_dt,
                             *(x.float() for x in (task.placed, self.held_any, task.lifted, task.entered,
                                                   task.toppled, task.thrown, auto_any)), self.min_grip, task.min_dist], -1)
        self.finished = (stats, done)
        self.last_next = (next_obs, next_state)
        return dict(actor_observation=obs, observation=state, action=action, reward=reward,
                    actor_next_observation=next_obs, next_observation=next_state, terminated=done.float())


# Reach shaping: the right gripper base sits ~8.5 cm from the can's centre when it holds the can
# (CLOSE / LIFT starts); REACH starts are ~33 cm and APPROACH ~63 cm away.
HELD_DISTANCE = 0.085
REACH_SCALES = (0.1, 0.5)  # m, exponential length scales of the reach potential (weights 0.5, 0.5)
HELD_PHASES = ("LIFT", "TURN_WALK", "PLACE_REACH", "LOWER", "RELEASE")  # recorded with the can in hand


def gather(value, world):
    if world == 1:
        return value
    merged = value.new_empty((world * len(value), *value.shape[1:]))
    dist.all_gather_into_tensor(merged, value.contiguous())
    return merged


def gpu_utilization(device):
    """Instantaneous utilization (%) of this rank's GPU, via pynvml or nvidia-smi; None if neither works."""
    try:
        return torch.cuda.utilization(device)
    except Exception:
        pass
    try:
        import subprocess

        uuid = str(torch.cuda.get_device_properties(device).uuid)
        uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", uuid],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip()) if out.returncode == 0 else None
    except Exception:
        return None


def atomic_save(value, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state(device):
    state = dict(cpu=torch.get_rng_state())
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def save_state(args, step, rank, world, device, learner, replay, task):
    # Per-rank collector state first; the learner file names the step every rank file must match.
    collector = dict(step=step, world=world, rng=rng_state(device), generator=task.generator.get_state())
    if args.save_replay:
        collector["replay"] = replay.state_dict()
    atomic_save(collector, args.out_dir / f"collector-rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        ckpt = dict(**learner.checkpoint(), step=step, decisions=step * world * task.n, world=world, num_envs=task.n,
                    noise_scale=args.noise_scale, noise_repeat=args.noise_repeat, upright_release=args.upright_release,
                    throw_speed=args.throw_speed, sampler=task.sampler.state_dict(),
                    episode_seconds=args.episode_seconds, reach_weight=args.reach_weight, dense_starts=args.dense_starts, gripper_hysteresis=args.gripper_hysteresis or None, measured_gripper=args.measured_gripper,
                    fm=str(args.fm), tracker_release=str(args.tracker_release))
        atomic_save(ckpt, args.out_dir / "flashsac.pt")
        if args.snapshot_every and step % args.snapshot_every == 0:
            atomic_save(ckpt, args.out_dir / f"flashsac-{step:08d}.pt")


def load_state(path, args, rank, world, device, learner, replay, task):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    learner.load_checkpoint(ckpt)
    task.sampler.load_state_dict(ckpt["sampler"])
    step = ckpt["step"]
    collector_path = path.parent / f"collector-rank{rank}.pt"
    collector = None
    if ckpt["world"] == world and collector_path.exists():
        collector = torch.load(collector_path, map_location="cpu", weights_only=True)
        if collector["step"] != step or collector["world"] != world:
            collector = None
    # All ranks must agree, or replay sizes (and therefore collective calls) would diverge.
    complete = torch.tensor([collector is not None and (not args.save_replay or "replay" in collector)],
                            dtype=torch.int32, device=device)
    if world > 1:
        dist.all_reduce(complete, op=dist.ReduceOp.MIN)
    if complete.item():
        if "replay" in collector:
            replay.load_state_dict(collector["replay"])
        torch.set_rng_state(collector["rng"]["cpu"])
        if "cuda" in collector["rng"]:
            torch.cuda.set_rng_state(collector["rng"]["cuda"], device)
        task.generator.set_state(collector["generator"])
    elif rank == 0:
        print(f"Collector state for step {step} is incomplete; resuming with an empty replay", flush=True)
    if rank == 0:
        print(f"Resumed {path} at step {step}, {learner.updates} updates, replay {replay.size}/rank", flush=True)
    return step


def train(args, device, rank, world):
    from can_pnp.scene.config import CanPnpSceneCfg, HeadCameraCfg, SceneGeometryCfg
    from can_pnp.scene.env_cfg import CanPnpEnvCfgArgs, make_env
    from can_pnp.task_spec import TaskCfg
    from hskill.learners.critic import FEAT_DIM
    # The checkpoint catalog stores the frozen inference code beside each release.
    for source in (args.fm.parent / "inference_source", args.tracker_release / "source"):
        if source.is_dir():
            sys.path.insert(0, str(source.resolve()))
    from hskill.s0.official_five_batch import BatchedOfficialFive

    geometry = SceneGeometryCfg(can_mass=.1)
    scene = CanPnpSceneCfg(geometry=geometry, camera=HeadCameraCfg(width=320, height=240))
    seed = args.seed + rank
    env = make_env(CanPnpEnvCfgArgs(num_envs=args.num_envs, randomize=False, seed=seed, scene=scene,
                                    enable_time_out=False, nan_termination=True),
                   device=str(device))
    try:
        if args.quiet_linesearch:
            # The linesearch iteration-limit warning is a print baked into the captured kernels;
            # clearing its bit (and recapturing) leaves the dynamics unchanged.
            from mujoco_warp._src.types import OverflowType

            env.sim.wp_model.opt.warn_overflow = int(env.sim.wp_model.opt.warn_overflow) & ~int(OverflowType.LS_ITERATIONS)
            env.sim.create_graph()
        ctrl = BatchedOfficialFive(env, args.fm, release=args.tracker_release, fm_batch=args.fm_batch,
                                   gripper_hysteresis=tuple(args.gripper_hysteresis) if args.gripper_hysteresis else None)
        fm = ctrl.fm.eval().requires_grad_(False)
        task = FivePointTask(env, ctrl, geometry, args.episode_seconds, args.gamma, seed, args.seed,
                             phases=tuple(args.phases), slack=args.time_slack, margin=args.time_margin,
                             eval_envs=args.eval_envs, eval_offset=rank * args.eval_envs,
                             measured_gripper=args.measured_gripper, reach_weight=args.reach_weight,
                             dense_starts=args.dense_starts,
                             task_cfg=TaskCfg(upright_release=args.upright_release, throw_speed=args.throw_speed),
                             noise_scale=args.noise_scale, noise_repeat=args.noise_repeat)
        action_dim = (1 if args.noise_repeat else fm.horizon) * fm.action_dim
        # Every rank owns a learner replica (gradients are averaged) and a replay shard.
        torch.manual_seed(args.seed)
        learner = FlashSAC(fm.n_tokens * fm.n_emb, FEAT_DIM, action_dim, device, args.gamma,
                           noise_scale=args.noise_scale, target_std=args.target_std)
        torch.manual_seed(seed)
        # FM tokens are BF16 outputs (cast to FP32): storing them as BF16 is lossless.
        replay = Replay(args.capacity // world, {k: torch.bfloat16 for k in ("actor_observation", "actor_next_observation")})
        n = task.n
        train_mask, eval_mask = ~task.eval_mask, task.eval_mask
        n_train = int(train_mask.sum())
        if replay.capacity < n_train:
            raise ValueError("Each rank's replay shard must hold at least one step of all its worlds")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # This run's own latest checkpoint wins over --resume, so a restarted job keeps its progress.
        resume = args.resume
        if args.auto_resume and (args.out_dir / "flashsac.pt").exists():
            resume = args.out_dir / "flashsac.pt"
        start = load_state(resume, args, rank, world, device, learner, replay, task) if resume else 0
        writer = None
        if rank == 0 and args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            # purge_step drops points a resumed run is about to rewrite.
            writer = SummaryWriter(args.out_dir / "tb", purge_step=start + 1 if start else None)
        obs, state = task.reset()
        reset = torch.ones(n_train, dtype=torch.bool, device=device)
        # Evaluation sums: episodes, EPISODE_STATS, then validation-episode count and successes.
        eval_window = torch.zeros(3 + len(EPISODE_STATS), dtype=torch.float64, device=device)
        clock, clock_step = time.perf_counter(), start
        # Window sums: decisions, reward, episodes, then EPISODE_STATS.
        window = torch.zeros(3 + len(EPISODE_STATS), dtype=torch.float64, device=device)
        # Per start phase: episodes, then the PHASE_STATS sums.
        phase_window = torch.zeros(len(task.phase_names), 1 + len(PHASE_STATS), dtype=torch.float64, device=device)
        phase_columns = [EPISODE_STATS.index(k) for k in PHASE_STATS]
        collect_s = update_s = 0.
        metrics = {}  # the latest update's losses
        for step in range(start + 1, args.steps + 1):
            action = torch.empty(n, action_dim, device=device)
            if learner.updates == 0:
                # Before learning, sample the FM prior exactly (clamped to the action bound).
                action[train_mask] = torch.randn(n_train, action_dim, device=device).clamp(-args.noise_scale, args.noise_scale) / args.noise_scale
            else:
                action[train_mask] = learner.act(obs[train_mask], reset)
            if eval_mask.any():
                action[eval_mask] = learner.act_deterministic(obs[eval_mask])
            tick = time.perf_counter()
            transition = task.step(obs, state, action, policy_controlled=learner.updates > 0)
            collect_s += time.perf_counter() - tick
            stats, done_all = task.finished
            evaluated = done_all & eval_mask
            if evaluated.any():
                eval_window[0] += evaluated.sum()
                eval_window[1:1 + len(EPISODE_STATS)] += stats[evaluated].sum(0)
                val = task.eval_val[task.row[evaluated.cpu()]].to(device)
                eval_window[-2] += val.sum()
                eval_window[-1] += (stats[evaluated][:, EPISODE_STATS.index("success")] * val).sum()
            transition = {k: v[train_mask] for k, v in transition.items()}
            stats, finished = stats[train_mask], done_all[train_mask]
            window[0] += n_train
            window[1] += transition["reward"].sum()
            window[2] += finished.sum()
            window[3:] += stats[finished].sum(0)
            ended = task.phase[train_mask][finished]
            phase_window[:, 0].index_add_(0, ended, torch.ones_like(ended, dtype=torch.float64))
            phase_window[:, 1:].index_add_(0, ended, stats[finished][:, phase_columns].double())
            replay.add(transition)
            outcome = gather(torch.stack((transition["reward"], transition["terminated"]), -1), world)
            done = outcome[:, 1].bool()
            learner.normalizer.update_reward_stats(outcome[:, 0], done, torch.zeros_like(done))
            reset = finished
            obs, state = task.last_next
            restart = done_all
            if restart.any():
                obs, state = obs.clone(), state.clone()
                obs[restart], state[restart] = task.reset(restart)
            # Shards grow by num_envs per step on every rank, so this branch agrees across ranks.
            if replay.size * world >= args.warmup:
                tick = time.perf_counter()
                if not learner.calibrated:
                    learner.calibrate(replay.sample(min(replay.size, args.calibration_size), device))
                for _ in range(args.updates_per_step):
                    # The actor and temperature update every second step: keep their latest values.
                    metrics.update(learner.update(replay.sample(args.batch_size, device)))
                update_s += time.perf_counter() - tick
            if step % args.log_every == 0 or step == args.steps:
                if world > 1:
                    dist.all_reduce(window)  # every rank reaches this step together
                    dist.all_reduce(phase_window)
                    dist.all_reduce(eval_window)
                if rank == 0:
                    now = time.perf_counter()
                    elapsed = max(now - clock, 1e-9)
                    row = dict(step=step, iteration=step, decisions=step * world * n,
                               decisions_per_s=round((step - clock_step) * world * n / elapsed, 2),
                               updates=learner.updates, replay=replay.size * world,
                               **{"time/collect_fraction": collect_s / elapsed,
                                  "time/update_fraction": update_s / elapsed,
                                  "reward/mean_per_decision": (window[1] / window[0]).item(),
                                  "episode/count": int(window[2].item())})
                    if window[2] > 0:
                        means = (window[3:] / window[2]).tolist()
                        row.update({f"episode/{k}": v for k, v in zip(EPISODE_STATS, means)})
                    if device.type == "cuda":
                        free, total = torch.cuda.mem_get_info(device)
                        row.update({"gpu/memory_used_gib": (total - free) / 2**30, "gpu/memory_total_gib": total / 2**30,
                                    "gpu/torch_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30})
                        utilization = gpu_utilization(device)
                        if utilization is not None:
                            row["gpu/utilization_percent"] = utilization
                    if eval_window[0] > 0:
                        row["eval/episodes"] = int(eval_window[0].item())
                        row.update({f"eval/{k}": (eval_window[1 + j] / eval_window[0]).item()
                                    for j, k in enumerate(EPISODE_STATS)})
                    if eval_window[-2] > 0:
                        row["eval/val_episodes"] = int(eval_window[-2].item())
                        row["eval/val_success"] = (eval_window[-1] / eval_window[-2]).item()
                    for i, name in enumerate(task.phase_names):
                        count = phase_window[i, 0].item()
                        if count > 0:
                            row[f"phase_{name}/episodes"] = int(count)
                            row.update({f"phase_{name}/{k}": (phase_window[i, 1 + j] / count).item()
                                        for j, k in enumerate(PHASE_STATS)})
                    for i, (success, share) in task.sampler.phase_metrics().items():
                        name = task.phase_names[i]
                        row[f"sgs/{name}_sampling_share"] = share
                        if success == success:
                            row[f"sgs/{name}_success"] = success
                    row.update(task.sampler.metrics())
                    row.update(metrics)
                    print(json.dumps(row), flush=True)
                    if writer is not None:
                        for key, value in row.items():
                            if key != "step":
                                writer.add_scalar(key if "/" in key else f"train/{key}", value, step)
                        writer.flush()
                    clock, clock_step = now, step
                window.zero_()
                phase_window.zero_()
                eval_window.zero_()
                collect_s = update_s = 0.
            if step % args.save_every == 0 or step == args.steps:
                save_state(args, step, rank, world, device, learner, replay, task)
        if writer is not None:
            writer.close()
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fm", type=Path, default=Path("downloads/5points_FM/fivepoints_fm.pt"))
    p.add_argument("--tracker-release", type=Path, default=Path("downloads/morphmimic-v0.1"))
    p.add_argument("--out-dir", type=Path, default=Path("logs/fivepoint_flashsac"))
    p.add_argument("--num-envs", type=int, default=32, help="Lockstep worlds per GPU")
    p.add_argument("--fm-batch", type=int, default=256, help="Worlds per FM forward pass (bounds GPU memory)")
    p.add_argument("--steps", type=int, default=20_000,
                   help="Iterations; each is one FM decision in every world of every GPU")
    p.add_argument("--warmup", type=int, default=2048, help="Global replay transitions before learning")
    p.add_argument("--capacity", type=int, default=32768, help="Global replay size, split evenly across ranks")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-GPU batch; the averaged gradient covers batch-size x GPUs samples")
    p.add_argument("--updates-per-step", type=int, default=1,
                   help="Synchronous updates per iteration (each adds num-envs decisions per GPU)")
    p.add_argument("--calibration-size", type=int, default=4096,
                   help="Per-rank replay samples for the first batch-norm statistics")
    p.add_argument("--target-std", type=float, default=.15,
                   help="Pre-tanh std defining the SAC target entropy (the FM prior is about 0.36)")
    p.add_argument("--gamma", type=float, default=.99, help="Discount per FM decision")
    p.add_argument("--dense-starts", action=argparse.BooleanOptionalAction, default=False,
                   help="Reset bank of every 0.5 s of each recorded episode up to RELEASE (labelled by phase) "
                        "instead of the phase starts only")
    p.add_argument("--reach-weight", type=float, default=1.0,
                   help="Weight of the potential-based reach shaping (gripper-to-can distance until the can "
                        "is first held; 0 disables). The can-to-goal shaping has weight 1")
    p.add_argument("--episode-seconds", type=float, default=30., help="Cap on every configuration's time budget")
    p.add_argument("--noise-scale", type=float, default=3.0,
                   help="FM latent bound: latent = noise-scale x tanh-bounded SAC action (DSRL's action magnitude)")
    p.add_argument("--noise-repeat", action=argparse.BooleanOptionalAction, default=False,
                   help="Steer one frame of FM latent repeated across the chunk (DSRL's pi0 parameterization) "
                        "instead of an independent latent per frame")
    p.add_argument("--upright-release", action=argparse.BooleanOptionalAction, default=False,
                   help="Success also needs the can upright and the right gripper commanded open")
    p.add_argument("--throw-speed", type=float, default=0.0,
                   help="m/s; > 0: a can not held and faster than this for 0.06 s fails the episode (thrown)")
    p.add_argument("--phases", nargs="+", default=list(PHASES), choices=PHASES,
                   help="Recorded start phases in the SGS reset bank (one configuration per phase and training episode)")
    p.add_argument("--time-slack", type=float, default=1.5,
                   help="Budget = slack x demonstration seconds from the start to RELEASE + margin")
    p.add_argument("--time-margin", type=float, default=4.0)
    p.add_argument("--gripper-hysteresis", type=float, nargs="*", default=[0.02, 0.05], metavar="M",
                   help="Right gripper closes below the first FM width and reopens above the second (m); "
                        "no values: raw FM widths")
    p.add_argument("--measured-gripper", action=argparse.BooleanOptionalAction, default=True,
                   help="Start held phases from the measured gripper opening rather than the commanded 0")
    p.add_argument("--eval-envs", type=int, default=0,
                   help="Worlds per GPU running the deterministic policy from every recorded APPROACH start in turn")
    p.add_argument("--quiet-linesearch", action=argparse.BooleanOptionalAction, default=True,
                   help="Silence MuJoCo Warp's per-step linesearch iteration-limit print")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
                   help="Rank zero also writes OUT_DIR/tb")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=10000, help="Also keep flashsac-STEP.pt (0 disables)")
    p.add_argument("--save-replay", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each rank's replay shard with the checkpoint for exact resumption")
    p.add_argument("--resume", type=Path,
                   help="Full flashsac.pt checkpoint to continue from (OUT_DIR/flashsac.pt wins under --auto-resume)")
    p.add_argument("--auto-resume", action="store_true", help="Continue from OUT_DIR/flashsac.pt when present")
    args = p.parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1))
    if (not 0 < args.gamma <= 1 or not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0
            or not (math.isfinite(args.reach_weight) and args.reach_weight >= 0)
            or not (math.isfinite(args.noise_scale) and args.noise_scale > 1)
            or not (math.isfinite(args.throw_speed) and args.throw_speed >= 0)
            or args.capacity < args.warmup or args.warmup < 2 * world or args.snapshot_every < 0
            or not 0 < args.target_std < 7 or args.time_slack <= 0 or args.time_margin < 0 or min(args.batch_size, args.steps, args.log_every,
                args.save_every, args.updates_per_step, args.calibration_size - 1, args.num_envs, args.fm_batch) < 1
            or not (len(args.gripper_hysteresis) == 0 or (len(args.gripper_hysteresis) == 2
                    and 0 <= args.gripper_hysteresis[0] <= args.gripper_hysteresis[1] <= 0.065))
            or not 0 <= args.eval_envs < args.num_envs
            or args.capacity // world < args.num_envs - args.eval_envs):
        p.error("Check positive counts, warmup >= 2 x GPUs, capacity >= max(warmup, GPUs x num-envs), episode length, "
                "target std and gamma in (0, 1]")
    if args.resume is not None and not args.resume.is_file():
        p.error(f"--resume {args.resume} does not exist")
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        train(args, device, dist.get_rank() if world > 1 else 0, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
