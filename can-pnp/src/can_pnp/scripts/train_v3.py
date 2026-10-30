"""S1-S0 v3: DSRL noise-policy GRPO mostly in imagination, with an object world model.

    uv run --extra diffusion python -m can_pnp.scripts.train_v3 --tracker-checkpoint <supermimic>.pt \\
        --diffusion-checkpoint <Humanoid-Diffusion run>/best.pt --out-dir logs/v3/run1 \\
        [--init-noise-checkpoint logs/dsrl/<run>/best.pt]
    # eight GPUs: scripts/train_v3_8gpu.sh

Frozen: DINOv3 + DiT (S1), the SuperMimic tracker and its body world model (S0). Trained: the
noise policy (``can_pnp.dsrl.NoisePolicy``) and the object world model
(``can_pnp.object_wm``). Every cycle (steps 1-2 only every ``collect_every`` cycles):

1. sim: one rollout of grasp-start episodes (8-step replans, a ``base_fraction`` of the envs on
   N(0, I) noise, failures kept) into the recency-weighted buffer; its policy envs measure the
   current noise policy (``sim/*``), and the object WM is scored on it before training on it
   (``wm_fresh/*``: its error on the newest policy's data). A rollout whose policy success
   falls more than ``rollback_margin`` below the best one restores that policy;
2. object WM: ``wm_steps`` steps on 32-step windows, conditioned on simulated or (for
   ``phi_fraction`` of them) Phi's gripper poses, labels always the simulated can; critic
   (``can_pnp.rl.critic``): ``critic_steps`` steps on the episode returns of recorded decisions;
3. ranking check (every ``rank_every`` cycles): candidates from identical sim states executed in
   sim vs imagined (``rank/tau``); a low tau halves the imagined iterations per sim rollout
   (more sim), a high tau doubles them, within [``imag_iters_min``, ``imag_iters_max``];
4. imagination: up to ``imag_iters`` GRPO updates, each on ``imag_starts`` recorded decisions x
   ``imag_group`` noises scored by ``can_pnp.rl.imagine.Imaginer`` (chunk returns of the sim task
   potential, see ``can_pnp.task``, completed by the critic's value of the chunk end), each
   stopped at ``imag_target_kl``, and all once the policy moved ``cycle_kl`` from the cycle's
   start (the simulator has the last word).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import tyro

from can_pnp.body.imagine import PhiStep
from can_pnp.body.kinematics import BodyKinematics
from can_pnp.body.world_model import DEFAULT_REPO, BodyWorldModel
from can_pnp.dsrl import NoisePolicy, SteeredDiffusionSource
from can_pnp.object_wm.buffer import EpisodeBuffer
from can_pnp.object_wm.model import ObjectWMCfg, ObjectWorldModel, object_loss
from can_pnp.rl.critic import ValueCritic, critic_features
from can_pnp.rl.grpo import Distributed, group_advantages, update
from can_pnp.rl.imagine import Imaginer
from can_pnp.sim.collect import SimCollector
from can_pnp.sim.rollouts import Rollouts


@dataclass
class Args:
    tracker_checkpoint: Path
    diffusion_checkpoint: Path
    diffusion_repo: Path = Path("../../Diffusion/Humanoid-Diffusion")
    supermimic_repo: Path = DEFAULT_REPO
    out_dir: Path = Path("logs/v3/run")
    resume: Path | None = None  # train_v3 checkpoint
    init_noise_checkpoint: Path | None = None  # start the noise policy from a train_dsrl checkpoint
    # Simulator (per GPU).
    num_envs: int = 512
    episode_s: float = 14.0
    grasp_phase: Literal["REACH", "CLOSE"] = "CLOSE"
    can_mass: float = 0.1
    base_fraction: float = 0.25  # collection envs on N(0, I) noise
    bf16: bool = True
    compile: bool = True
    rgb_backend: Literal["mujoco_warp", "hyperlux"] = "mujoco_warp"
    render_size: tuple[int, int] = (320, 240)
    # Task (as train_dsrl).
    success_reward: float = 10.0
    dist_scale: float = 0.1
    success_hold_s: float = 0.2
    upright_cos: float = 0.866
    release_opening: float = 0.03
    horizon: int = 32  # imagined frames per decision: the whole reference
    # Object world model.
    wm_hidden: tuple[int, ...] = (512, 512, 512)
    wm_lr: float = 3e-4
    wm_batch: int = 4096
    wm_warmup_rollouts: int = 2
    wm_warmup_steps: int = 2000
    wm_steps: int = 200
    phi_fraction: float = 0.5
    # Critic of the imagined chunk end (rest of the episode from a body + can state; 0 steps: greedy chunk return).
    critic_hidden: tuple[int, ...] = (512, 512, 512)
    critic_lr: float = 3e-4
    critic_batch: int = 4096
    critic_warmup_steps: int = 1000
    critic_steps: int = 200
    buffer_rollouts: int = 8
    buffer_decay: float = 0.7  # sampling weight per rollout of age
    # Imagined GRPO.
    imag_starts: int = 256
    imag_group: int = 16
    imag_iters: int = 8
    imag_iters_min: int = 1
    imag_iters_max: int = 32
    imag_batch: int = 1024  # DiT / world-model batch
    # Imagined updates of a cycle stop once the policy is this far (mean KL over recorded
    # decisions, summed over the noise dims) from where the cycle started: the simulator checks
    # the policy between cycles. A sim GRPO iteration moves it ~0.015.
    cycle_kl: float = 0.05
    # Simulator rollouts (+ object-WM training) only every collect_every cycles; the cycles between
    # are ranking check + imagination only. After a rollout whose policy success is more than
    # rollback_margin below the best one, the noise policy goes back to that best one and
    # cycle_kl halves (0 disables).
    collect_every: int = 4
    rollback_margin: float = 0.04
    lr: float = 2e-5
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    kl_coef: float = 1e-3
    target_kl: float = 0.05
    # Early stop of one imagined update: well below cycle_kl, so the budget takes a few updates.
    imag_target_kl: float = 0.01
    max_grad_norm: float = 1.0
    hidden: int = 512
    init_log_std: float = 0.0
    # Ranking check.
    rank_every: int = 1
    rank_group: int = 8
    rank_max_delay: int = 6  # decisions on shared noise before the candidates
    tau_low: float = 0.3
    tau_high: float = 0.6
    # Bookkeeping.
    cycles: int = 1000
    save_every: int = 5
    seed: int = 0


def gaussian_kl(mean: torch.Tensor, log_std, ref_mean: torch.Tensor, ref_log_std) -> torch.Tensor:
    """KL(N(mean, std) || N(ref_mean, ref_std)) per row, diagonal, summed over dims."""
    var, ref_var = torch.exp(2 * log_std), torch.exp(2 * ref_log_std)
    return 0.5 * ((var + (mean - ref_mean).square()) / ref_var - 1 + 2 * (ref_log_std - log_std)).sum(-1)


def _mean(stats: list[dict]) -> dict:
    return {k: sum(float(s[k]) for s in stats) / len(stats) for k in stats[0]} if stats else {}


def _all_mean(stats: dict, ddp: Distributed, device) -> dict:
    """Average the float stats over ranks (NaN-free entries only)."""
    keys = sorted(k for k, v in stats.items() if isinstance(v, float))
    if not ddp.enabled or not keys:
        return stats
    v = torch.tensor([stats[k] for k in keys], device=device)
    finite = v.isfinite().float()
    total = ddp.sum(torch.stack([torch.nan_to_num(v), finite]))
    out = dict(stats)
    for k, s, c in zip(keys, total[0].tolist(), total[1].tolist()):
        out[k] = s / c if c else float("nan")
    return out


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    ddp = Distributed()
    device = ddp.device
    torch.manual_seed(args.seed + 1000 * ddp.rank)
    rollouts = Rollouts(args, device, ddp.rank, num_envs=args.num_envs)
    env, diffusion = rollouts.env, rollouts.diffusion
    body = BodyWorldModel(env, args.tracker_checkpoint, args.supermimic_repo)
    kin = BodyKinematics(env, body, compile=args.compile)
    phi = PhiStep(body, rollouts.tracker, compile=args.compile)
    collector = SimCollector(rollouts, phi, kin, args.horizon)
    noise_policy = NoisePolicy(diffusion.n_tokens, diffusion.n_emb, diffusion.horizon, diffusion.action_dim,
                               hidden=args.hidden, init_log_std=args.init_log_std).to(device)  # fmt: skip
    optimizer = torch.optim.Adam(noise_policy.parameters(), lr=args.lr)
    object_wm = ObjectWorldModel(ObjectWMCfg(hidden=args.wm_hidden, dt=env.step_dt)).to(device)
    if args.compile:
        object_wm.compile_step()
    wm_optimizer = torch.optim.Adam(object_wm.parameters(), lr=args.wm_lr)
    critic = ValueCritic(args.critic_hidden).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    use_critic = args.critic_steps > 0
    decoder = SteeredDiffusionSource(diffusion, noise_policy, bf16=args.bf16)
    imaginer = Imaginer(decoder, phi, kin, object_wm, rollouts.task, rollouts.geo, args.horizon, args.imag_batch,
                        critic if use_critic else None)  # fmt: skip
    imag_cfg = dataclasses.replace(args, target_kl=args.imag_target_kl)
    buffer = EpisodeBuffer(args.buffer_rollouts, args.buffer_decay, args.horizon)
    start, imag_iters = 0, args.imag_iters
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        noise_policy.load_state_dict(state["noise_policy"])
        optimizer.load_state_dict(state["optimizer"])
        object_wm.load_state_dict(state["object_wm"])
        wm_optimizer.load_state_dict(state["wm_optimizer"])
        if "critic" in state:
            critic.load_state_dict(state["critic"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        start, imag_iters = state["cycle"] + 1, state["imag_iters"]
        args.cycle_kl = state.get("cycle_kl", args.cycle_kl)
    elif args.init_noise_checkpoint is not None:
        # Not its optimizer: Adam's second moments of the large sim batches would blow up the
        # steps on the (smaller, noisier) imagined batches.
        noise_policy.load_state_dict(torch.load(args.init_noise_checkpoint, map_location=device,
                                                weights_only=False)["noise_policy"])  # fmt: skip
    ddp.broadcast(noise_policy)
    ddp.broadcast(object_wm)
    ddp.broadcast(critic)
    log = None
    if ddp.rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "args.json").write_text(json.dumps(asdict(args), default=str, indent=2))
        log = (args.out_dir / "metrics.jsonl").open("a")
        print(f"v3: {ddp.world} GPU x {args.num_envs} sim envs, imagination {args.imag_starts} x {args.imag_group}",
              flush=True)  # fmt: skip
    rng = torch.Generator().manual_seed(args.seed + 7919 * ddp.rank)
    train_rows = torch.tensor(rollouts.train_rows)
    n = env.num_envs

    def collect():
        rows = train_rows[torch.randint(len(train_rows), (n,), generator=rng)]
        base = torch.rand(n, generator=rng) < args.base_fraction
        return collector.collect(noise_policy, rows, base.to(device))

    def wm_batch_loss(source: EpisodeBuffer, size: int, phi_fraction: float):
        b = source.windows(size, phi_fraction)
        pred, logits = object_wm.rollout(b["obj"][0], b["grip"], b["command"], b["box"], b["table"])
        return object_loss(pred, logits, b["obj"], b["held"], b["mask"])

    def train_wm(steps: int) -> dict:
        object_wm.train()
        metrics = []
        for _ in range(steps):
            loss, m = wm_batch_loss(buffer, args.wm_batch, args.phi_fraction)
            wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ddp.average_grads(object_wm)
            grad = torch.nn.utils.clip_grad_norm_(object_wm.parameters(), 10.0)
            if bool(ddp.sum((~(loss.isfinite() & grad.isfinite())).float().reshape(1)) > 0):
                continue  # a non-finite step on any rank skips it everywhere
            wm_optimizer.step()
            metrics.append({"loss": loss.detach(), **m})
        object_wm.eval()
        return {f"wm/{k}": v for k, v in _mean(metrics[-20:]).items()}

    def critic_batch_loss(source: EpisodeBuffer, size: int):
        b = source.starts(size, policy_only=True)
        feat = critic_features(body, b["s0"], b["obj0"], b["box"], b["table"], b["goal"], b["time"])
        pred = critic(feat)
        # Remaining change of the task potential: the episode's return minus the decision's potential.
        target = b["final"] - torch.exp(-(b["obj0"][:, :3] - b["goal"]).norm(dim=-1) / rollouts.task.dist_scale)
        loss = torch.nn.functional.mse_loss(pred, target)
        with torch.no_grad():
            explained = 1 - (pred - target).var() / target.var().clamp(min=1e-8)
        return loss, {"explained_var": explained, "value_mean": pred.mean().detach(), "target_mean": target.mean()}

    def train_critic(steps: int) -> dict:
        if not use_critic or not any(len(v) for v in buffer.valid_policy):
            return {}
        critic.train()
        metrics = []
        for _ in range(steps):
            loss, m = critic_batch_loss(buffer, args.critic_batch)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ddp.average_grads(critic)
            grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            if bool(ddp.sum((~(loss.isfinite() & grad.isfinite())).float().reshape(1)) > 0):
                continue
            critic_optimizer.step()
            metrics.append({"loss": loss.detach(), **m})
        critic.eval()
        return {f"critic/{k}": v for k, v in _mean(metrics[-20:]).items()}

    @torch.no_grad()
    def critic_fresh(episodes) -> dict:
        fresh = EpisodeBuffer(1, 1.0, args.horizon)
        fresh.add(episodes)
        if not use_critic or not len(fresh.valid_policy[0]):
            return {}
        loss, m = critic_batch_loss(fresh, args.critic_batch)
        return {"critic_fresh/loss": float(loss), "critic_fresh/explained_var": float(m["explained_var"])}

    @torch.no_grad()
    def wm_fresh(episodes) -> dict:
        fresh = EpisodeBuffer(1, 1.0, args.horizon)
        fresh.add(episodes)
        if not len(fresh.valid[0]):
            return {}
        out = {}
        for name, phi in (("sim_body", 0.0), ("phi_body", 1.0)):
            loss, m = wm_batch_loss(fresh, args.wm_batch, phi)
            out.update({f"wm_fresh/{name}_{k}": float(v) for k, v in {"loss": loss, **m}.items()
                        if k in ("loss", "pos_err_cm_final", "pos_err_cm_mean", "held_acc")})  # fmt: skip
        return out

    def imagine_update() -> dict:
        starts = buffer.starts(args.imag_starts)
        rep = {k: v.repeat_interleave(args.imag_group, dim=0) for k, v in starts.items()}
        with torch.no_grad():
            mean, log_std = noise_policy(rep["cond"])
            w = mean + torch.exp(log_std) * torch.randn_like(mean)
            log_prob = NoisePolicy.log_prob(w, mean, log_std)
            out = imaginer.chunk_return(rep, w)
        returns = out["returns"]
        # Far from the goal the potential exp(-d / dist_scale) and its changes are tiny: candidates
        # within 0.1% of the start potential count as ties (no ranking signal).
        min_std = 1e-3 * out["potential0"].view(-1, args.imag_group)[:, :1]
        adv, informative = group_advantages(returns, args.imag_group, min_std)
        batch = {"cond": rep["cond"][None], "noise": w[None], "log_prob": log_prob[None],
                 "mask": returns.isfinite().float()[None]}  # fmt: skip
        stats = update(noise_policy, optimizer, batch, torch.nan_to_num(adv), imag_cfg, ddp)
        return {
            "imag/return": float(returns.mean()), "imag/placed": float(out["placed"].float().mean()),
            "imag/dropped": float(out["dropped"].float().mean()), "imag/final_dist": float(out["final_dist"].mean()),
            "imag/fall_prob": float(out["fall_prob"].mean()), "imag/informative_groups": float(informative.mean()),
            "imag/value": float(out["value"].mean()),
            **{f"grpo/{k}": float(v) for k, v in stats.items()},
        }  # fmt: skip

    # The buffer is not checkpointed: fill it first (and warm the object WM up on a new run).
    for _ in range(args.wm_warmup_rollouts):
        episodes, _ = collect()
        buffer.add(episodes)
    if args.resume is None:
        warm = train_wm(args.wm_warmup_steps)
        warm.update(train_critic(args.critic_warmup_steps))
        if ddp.rank == 0:
            print("object WM / critic warm-up:", {k: round(v, 4) for k, v in warm.items()}, flush=True)
    best = -float("inf")
    best_success, best_policy = -1.0, None
    for cycle in range(start, args.cycles):
        tic = time.perf_counter()
        sim_stats, fresh, wm_stats, control = {}, {}, {}, {}
        if (cycle - start) % args.collect_every == 0:
            episodes, sim_stats = collect()
            fresh = {**wm_fresh(episodes), **critic_fresh(episodes)}
            buffer.add(episodes)
            del episodes
            sim_stats = _all_mean(sim_stats, ddp, device)
            success = sim_stats.get("sim/success_rate", float("nan"))
            if success > best_success:
                best_success = success
                best_policy = copy.deepcopy((noise_policy.state_dict(), optimizer.state_dict()))
            elif args.rollback_margin > 0 and best_success - success > args.rollback_margin:
                noise_policy.load_state_dict(best_policy[0])
                optimizer.load_state_dict(best_policy[1])
                args.cycle_kl *= 0.5
                control["rollback"] = 1.0
            t_sim = time.perf_counter()
            wm_stats = {**train_wm(args.wm_steps), **train_critic(args.critic_steps)}
        else:
            t_sim = time.perf_counter()
        t_wm = time.perf_counter()
        rank = {}
        if args.rank_every and cycle % args.rank_every == 0:
            g = n // args.rank_group
            rows = train_rows[torch.randint(len(train_rows), (g,), generator=rng)]
            delay = int(torch.randint(args.rank_max_delay + 1, (1,), generator=rng))
            rank = _all_mean(collector.ranking(noise_policy, imaginer, rows, args.rank_group, delay), ddp, device)
            tau = rank["rank/tau"]
            if not math.isnan(tau):
                if tau < args.tau_low:
                    imag_iters = max(args.imag_iters_min, imag_iters // 2)
                elif tau > args.tau_high:
                    imag_iters = min(args.imag_iters_max, imag_iters * 2)
        t_rank = time.perf_counter()
        probe = buffer.starts(args.imag_starts)["cond"]
        with torch.no_grad():
            ref_mean, ref_log_std = noise_policy(probe)
        imag_stats, drift = [], 0.0
        for _ in range(imag_iters):
            imag_stats.append(imagine_update())
            with torch.no_grad():
                mean, log_std = noise_policy(probe)
                kl = gaussian_kl(mean, log_std, ref_mean, ref_log_std).mean().reshape(1)
            drift = float(ddp.sum(kl) / ddp.world)
            if drift > args.cycle_kl:
                break
        imag = {**_mean(imag_stats), "imag/updates": float(len(imag_stats)), "imag/cycle_kl": drift}
        t_imag = time.perf_counter()
        record = {
            "cycle": cycle, "imag_iters": imag_iters, "cycle_kl": args.cycle_kl, **control,
            "sim_s": t_sim - tic, "wm_s": t_wm - t_sim,
            "rank_s": t_rank - t_wm, "imag_s": t_imag - t_rank,
            "std": float(noise_policy.log_std.detach().clamp_min(noise_policy.min_log_std).exp().mean()),
            **_all_mean({**sim_stats, **fresh, **wm_stats, **imag}, ddp, device), **rank,
        }  # fmt: skip
        if ddp.rank == 0:
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            state = {
                "noise_policy": noise_policy.state_dict(), "optimizer": optimizer.state_dict(),
                "object_wm": object_wm.state_dict(), "wm_optimizer": wm_optimizer.state_dict(),
                "critic": critic.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                "cycle": cycle, "imag_iters": imag_iters, "cycle_kl": args.cycle_kl, "args": asdict(args),
                "diffusion_checkpoint": str(args.diffusion_checkpoint),
                "tracker_checkpoint": str(args.tracker_checkpoint),
            }  # fmt: skip
            params = list(state["noise_policy"].values()) + list(state["object_wm"].values())
            if not all(bool(v.isfinite().all()) for v in params if v.is_floating_point()):
                raise RuntimeError(f"Non-finite parameters at cycle {cycle}; last.pt kept")
            if cycle % args.save_every == 0 or cycle == args.cycles - 1:
                torch.save(state, args.out_dir / "last.pt")
            score = record.get("sim/return", -float("inf"))
            if score > best:
                best = score
                torch.save(state, args.out_dir / "best.pt")
    if ddp.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
