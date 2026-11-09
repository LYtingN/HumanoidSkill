"""S1-S0 v3: DSRL noise-policy GRPO in simulation and imagination, with cross-layer attribution.

See docs/ATTRIBUTION_ARCHITECTURE.md for the architecture.

    uv run --extra diffusion python -m experiments.train_v3 --tracker-checkpoint <supermimic>.pt \\
        --diffusion-checkpoint <Humanoid-Diffusion run>/best.pt --out-dir logs/v3/run1 \\
        [--init-noise-checkpoint logs/dsrl/<run>/best.pt]
    # eight GPUs: scripts/train_v3_8gpu.sh

Frozen: DINOv3 + DiT (S1), the SuperMimic tracker and its body world model (S0). Trained: the
noise policy (``hskill.s1.noise_policy.NoisePolicy``) and the object world model
(``hskill.interaction``). Every cycle (steps 1-2 only every ``collect_every`` cycles):

1. sim: one rollout of grasp-start episodes in groups of ``sim_group`` envs per start (8-step
   replans, a ``base_fraction`` of the groups on N(0, I) noise, failures kept) into the
   recency-weighted buffer; its policy envs measure the current noise policy (``sim/*``) and
   how exactly S0 executes the references (``track/*``: right hand vs the reference's hand on the
   actual body), the object WM is scored on it before training on it (``wm_fresh/*``), and with
   ``sim_anchor`` the noise policy takes one GRPO update on the episodes' own returns (the
   simulator's signal: never worse than DSRL). A rollout whose policy success falls more than
   ``rollback_margin`` below the best one restores that policy;
2. object WM: ``wm_steps`` steps on 32-step windows, conditioned on simulated or (for
   ``phi_fraction`` of them) Phi's gripper poses, labels always the simulated can; critic
   (``hskill.learners.critic``): ``critic_steps`` steps on the episode returns of recorded decisions;
3. ranking check (every ``rank_every`` cycles): candidates from identical sim states executed in
   sim vs imagined (``rank/tau``), and the ideal branch against the candidates S0 executed
   exactly (``rank/ideal_*``, ``rank/gap_*``); below ``tau_min`` the cycle skips imagination and the
   next one collects (the imagined ranking is no better than chance); a low tau halves the
   imagined iterations per sim rollout, a high tau doubles them, within
   [``imag_iters_min``, ``imag_iters_max``];
4. imagination: up to ``imag_iters`` GRPO updates, each on ``imag_starts`` recorded decisions x
   ``imag_group`` noises scored by ``hskill.executors.imaginer.Imaginer`` (chunk returns of the sim task
   potential, see ``can_pnp.task_spec``, completed by the critic's value of the chunk end) on the
   actual branch J, plus ``intent_weight`` x the ideal branch's excess max(J* - J, 0) (0 while S0
   stays frozen: an intent S0 cannot execute is worth what S0 achieves); candidates the object WM
   ensemble disagrees on by more than ``uncertainty_max`` get no advantage; each
   stopped at ``imag_target_kl``, and all once the policy moved ``cycle_kl`` from the cycle's
   start (the simulator has the last word).
"""

from __future__ import annotations

import contextlib
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
from can_pnp.task_spec import TaskTracker
from hskill.data.buffer import TIME_MAJOR, EpisodeBuffer
from hskill.data.demo_progress import PROPRIO_DIM, DemoProgress
from hskill.embodiment.g1_aloha import BODY_JOINT_NAMES
from hskill.executors.imagine import PhiStep
from hskill.executors.imaginer import Imaginer
from hskill.interaction.model import ObjectWMCfg, ObjectWMEnsemble, ObjectWorldModel, object_loss
from hskill.learners.critic import (
    ProgressCritic,
    ValueCritic,
    critic_features,
    object_features,
    progress_features,
    state_proprio,
)
from hskill.learners.grpo import Distributed, group_advantages, masked_group_advantages, update
from hskill.s0.body_world_model import DEFAULT_REPO, BodyWorldModel
from hskill.s0.kinematics import BodyKinematics
from hskill.s1.noise_policy import NoisePolicy, SteeredDiffusionSource
from hskill.sim.collect import SimCollector
from hskill.sim.rollouts import Rollouts

from experiments._task_setup import rollout_dependencies


@dataclass
class Args:
    tracker_checkpoint: Path
    diffusion_checkpoint: Path
    diffusion_repo: Path = Path(__file__).resolve().parents[2] / "Diffusion" / "Humanoid-Diffusion"
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
    wm_ensemble: int = 3  # object WM members; their disagreement gates the imagined candidates
    uncertainty_max: float = 0.03  # m: final object position spread beyond which a candidate is not trusted
    # Critic of the imagined chunk end (rest of the episode from a body + can state; 0 steps: greedy chunk return).
    critic_hidden: tuple[int, ...] = (512, 512, 512)
    critic_lr: float = 3e-4
    critic_batch: int = 4096
    critic_warmup_steps: int = 1000
    critic_steps: int = 200
    # progress: W(state) = discounted final potential, pre-trained on the demonstrations' time to
    # completion (proprioception only; as RISE) and fine-tuned on the policy's episodes;
    # residual: the remaining potential change from body + object state, rollouts only.
    critic_kind: Literal["progress", "residual"] = "progress"
    critic_gamma: float = 0.997  # per control step
    demo_root: Path = Path(__file__).resolve().parents[1] / "data" / "residual_rl" / "can_pnp"
    demo_fraction: float = 0.5  # share of demonstration samples in a critic batch
    critic_demo_steps: int = 2000  # demonstration-only pre-training of a new run
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
    sim_group: int = 16  # collection envs per grasp start (the sim GRPO groups)
    sim_anchor: bool = True  # one GRPO update per collection on the episodes' own returns
    well_tracked: float = 0.02  # m: right-hand error of an exactly executed step
    # Weight of the ideal branch's excess over the actual one in the imagined score (intent loss).
    # Keep 0 while S0 is frozen.
    intent_weight: float = 0.0
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
    tau_min: float = 0.2  # below it: no imagined updates, next cycle collects
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
    rollouts = Rollouts(args, device, ddp.rank, num_envs=args.num_envs, **rollout_dependencies(args))
    env, diffusion = rollouts.env, rollouts.diffusion
    body = BodyWorldModel(env, args.tracker_checkpoint, args.supermimic_repo)
    kin = BodyKinematics(env, body, compile=args.compile)
    phi = PhiStep(body, rollouts.tracker, compile=args.compile)
    collector = SimCollector(rollouts, phi, kin, args.horizon, args.well_tracked)
    noise_policy = NoisePolicy(diffusion.n_tokens, diffusion.n_emb, diffusion.horizon, diffusion.action_dim,
                               hidden=args.hidden, init_log_std=args.init_log_std).to(device)  # fmt: skip
    optimizer = torch.optim.Adam(noise_policy.parameters(), lr=args.lr)
    wm_cfg = ObjectWMCfg(hidden=args.wm_hidden, dt=env.step_dt)
    object_wm = (ObjectWMEnsemble(wm_cfg, args.wm_ensemble) if args.wm_ensemble > 1 else ObjectWorldModel(wm_cfg)).to(device)
    if args.compile:
        object_wm.compile_step()
    wm_optimizer = torch.optim.Adam(object_wm.parameters(), lr=args.wm_lr)
    critic = (ProgressCritic(args.critic_hidden) if args.critic_kind == "progress" else ValueCritic(args.critic_hidden)).to(device)
    demos = None
    if args.critic_kind == "progress" and args.critic_steps > 0:
        starts = rollouts.starts
        frames = {starts.episode_ids[r]: int(starts.frame[r]) for r in rollouts.train_rows}
        demos = DemoProgress(args.demo_root, frames, BODY_JOINT_NAMES, args.critic_gamma, rollouts.task.success_reward,
                             round(args.episode_s / env.step_dt), device)  # fmt: skip
        if ddp.rank == 0:
            print(f"critic: {demos.episodes} demonstrations, {len(demos)} frames from the {args.grasp_phase} starts", flush=True)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    use_critic = args.critic_steps > 0
    decoder = SteeredDiffusionSource(diffusion, noise_policy, bf16=args.bf16)
    imaginer = Imaginer(decoder, phi, kin, object_wm, rollouts.task, rollouts.geo, args.horizon, args.imag_batch,
                        critic if use_critic else None, task_tracker=TaskTracker)  # fmt: skip
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

    if n % args.sim_group:
        raise ValueError(f"{n} envs do not split into groups of {args.sim_group}")
    groups = n // args.sim_group

    def collect():
        rows = train_rows[torch.randint(len(train_rows), (groups,), generator=rng)].repeat_interleave(args.sim_group)
        base = (torch.rand(groups, generator=rng) < args.base_fraction).repeat_interleave(args.sim_group)
        return collector.collect(noise_policy, rows, base.to(device))

    def sim_anchor(episodes) -> dict:
        """One GRPO update on the policy groups' episode returns (DSRL's update)."""
        returns = episodes.final_potential
        adv, informative = group_advantages(returns, args.sim_group)
        mask = (episodes.decision_live & ~episodes.base[None]).float()
        batch = {"cond": episodes.cond, "noise": episodes.noise, "log_prob": episodes.log_prob, "mask": mask}
        stats = update(noise_policy, optimizer, batch, adv, args, ddp)
        return {"anchor/informative_groups": float(informative.mean()), **{f"anchor/{k}": float(v) for k, v in stats.items()}}

    def wm_batch_loss(source: EpisodeBuffer, size: int, phi_fraction: float):
        b = source.windows(size, phi_fraction)
        pred, logits = object_wm.rollout(b["obj"][0], b["grip"], b["command"], b["box"], b["table"])
        return object_loss(pred, logits, b["obj"], b["held"], b["mask"])

    def wm_members_loss(size: int):
        """Every ensemble member on its own batch (one stacked rollout): the members' spread then
        reflects the data's coverage."""
        if not isinstance(object_wm, ObjectWMEnsemble):
            return wm_batch_loss(buffer, size, args.phi_fraction)
        E = object_wm.num_members
        b = buffer.windows(E * size, args.phi_fraction)
        H = b["command"].shape[0]
        # The windows come grouped by episode: shuffle before splitting them among the members.
        perm = torch.randperm(E * size, device=device)
        b = {k: v[:, perm] if k in TIME_MAJOR else v[perm] for k, v in b.items()}
        objs, logits = object_wm.rollout_members(b["obj"][0].view(E, size, -1), b["grip"].view(H + 1, E, size, -1),
                                                 b["command"].view(H, E, size), b["box"].view(E, size, 3),
                                                 b["table"].view(E, size, 3))  # fmt: skip
        pred = objs.transpose(0, 1).reshape(H + 1, E * size, -1)
        return object_loss(pred, logits.transpose(0, 1).reshape(H, E * size), b["obj"], b["held"], b["mask"])

    @contextlib.contextmanager
    def tf32():
        """TF32 matrix products while fitting the object WM and the critic (their fp32 matmuls
        are most of that time); imagination and the policies keep full fp32."""
        matmul = torch.backends.cuda.matmul
        prev, matmul.fp32_precision = matmul.fp32_precision, "tf32"
        try:
            yield
        finally:
            matmul.fp32_precision = prev

    @tf32()
    def train_wm(steps: int) -> dict:
        object_wm.train()
        metrics = []
        for _ in range(steps):
            loss, m = wm_members_loss(args.wm_batch)
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

    def progress_batch(source: EpisodeBuffer | None, size: int) -> tuple[list, list, list]:
        """Demonstration and rollout samples of the progress critic (features, targets, names)."""
        feats, targets, names = [], [], []
        n_demo = size if source is None else round(size * args.demo_fraction)
        if demos is not None and n_demo:
            d = demos.sample(n_demo)
            feats.append(progress_features(d["proprio"], None))
            targets.append(d["target"])
            names.append("demo")
        if source is not None and any(len(v) for v in source.valid_policy) and size - n_demo > 0:
            b = source.starts(size - n_demo, policy_only=True)
            objects = object_features(b["s0"], b["obj0"], b["box"], b["table"], b["goal"])
            feats.append(progress_features(state_proprio(kin, b["s0"], b["obj0"][:, 13]), objects))
            targets.append(b["final"] * args.critic_gamma ** b["to_end"])
            names.append("sim")
        return feats, targets, names

    def critic_batch_loss(source: EpisodeBuffer | None, size: int):
        if args.critic_kind == "progress":
            feats, targets, names = progress_batch(source, size)
            loss, metrics = 0.0, {}
            for f, y, name in zip(feats, targets, names):
                pred = critic(f)
                part = torch.nn.functional.mse_loss(pred, y)
                loss = loss + part * len(y) / size
                with torch.no_grad():
                    metrics[f"{name}_explained_var"] = 1 - (pred - y).var() / y.var().clamp(min=1e-4)
                    metrics[f"{name}_value_mean"] = pred.mean()
            return loss, metrics
        b = source.starts(size, policy_only=True)
        feat = critic_features(body, b["s0"], b["obj0"], b["box"], b["table"], b["goal"], b["time"])
        pred = critic(feat)
        # Remaining change of the task potential: the episode's return minus the decision's potential.
        target = b["final"] - torch.exp(-(b["obj0"][:, :3] - b["goal"]).norm(dim=-1) / rollouts.task.dist_scale)
        loss = torch.nn.functional.mse_loss(pred, target)
        with torch.no_grad():
            explained = 1 - (pred - target).var() / target.var().clamp(min=1e-8)
        return loss, {"explained_var": explained, "value_mean": pred.mean().detach(), "target_mean": target.mean()}

    @tf32()
    def train_critic(steps: int, source: EpisodeBuffer | None = None) -> dict:
        if not use_critic or (demos is None and (source is None or not any(len(v) for v in source.valid_policy))):
            return {}
        critic.train()
        metrics = []
        for _ in range(steps):
            loss, m = critic_batch_loss(source, args.critic_batch)
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
        if args.critic_kind == "progress":
            feats, targets, names = progress_batch(fresh, 2 * args.critic_batch)  # half demos, half the new rollout
            out = {}
            for f, y, name in zip(feats, targets, names):
                pred = critic(f)
                out[f"critic_fresh/{name}_explained_var"] = float(1 - (pred - y).var() / y.var().clamp(min=1e-4))
                if name == "sim":
                    # The progress imagination uses (proprioception only) against the real outcomes.
                    own = f.clone()
                    own[:, PROPRIO_DIM:] = 0.0
                    progress = critic(own)
                    out["critic_fresh/sim_progress_mean"] = float(progress.mean())
                    out["critic_fresh/sim_progress_corr"] = float(torch.corrcoef(torch.stack([progress, y]))[0, 1])
            return out
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
        returns, ideal = out["returns"], out["returns_ideal"]
        gap = ideal - returns
        score = returns + args.intent_weight * gap.clamp(min=0)
        keep = score.isfinite() & (out["uncertainty"] <= args.uncertainty_max)
        if args.intent_weight > 0:
            keep &= out["uncertainty_ideal"] <= args.uncertainty_max
        # Far from the goal the potential exp(-d / dist_scale) and its changes are tiny: candidates
        # within 0.1% of the start potential count as ties (no ranking signal).
        min_std = 1e-3 * out["potential0"].view(-1, args.imag_group)[:, :1]
        adv, informative = masked_group_advantages(torch.nan_to_num(score), args.imag_group, keep, min_std)
        batch = {"cond": rep["cond"][None], "noise": w[None], "log_prob": log_prob[None], "mask": keep.float()[None]}
        stats = update(noise_policy, optimizer, batch, adv, imag_cfg, ddp)
        trusted_ideal = out["uncertainty_ideal"] <= args.uncertainty_max
        return {
            "imag/return_ideal": float(ideal.mean()), "imag/gap": float(gap.mean()),
            "imag/closable_share": float(((gap > 0.1) & trusted_ideal).float().mean()),
            "imag/uncertainty_cm": float(100 * out["uncertainty"].mean()),
            "imag/uncertainty_ideal_cm": float(100 * out["uncertainty_ideal"].mean()),
            "imag/kept_share": float(keep.float().mean()),
            "imag/return": float(returns.mean()), "imag/placed": float(out["placed"].float().mean()),
            "imag/dropped": float(out["dropped"].float().mean()), "imag/final_dist": float(out["final_dist"].mean()),
            "imag/fall_prob": float(out["fall_prob"].mean()), "imag/informative_groups": float(informative.mean()),
            "imag/value": float(out["value"].mean()),
            # How much the candidates of a start differ: by the critic, and by the chunk outcome.
            "imag/value_group_std": float(out["value"].view(-1, args.imag_group).std(1).nanmean()),
            "imag/chunk_return_group_std": float(out["chunk_returns"].view(-1, args.imag_group).std(1).nanmean()),
            **{f"grpo/{k}": float(v) for k, v in stats.items()},
        }  # fmt: skip

    if args.resume is None and demos is not None:
        pre = train_critic(args.critic_demo_steps)  # demonstrations only, before any rollout
        if ddp.rank == 0:
            print("critic pre-training on demonstrations:", {k: round(v, 4) for k, v in pre.items()}, flush=True)
    # The buffer is not checkpointed: fill it first (and warm the object WM up on a new run).
    for _ in range(args.wm_warmup_rollouts):
        episodes, _ = collect()
        buffer.add(episodes)
    if args.resume is None:
        warm = train_wm(args.wm_warmup_steps)
        warm.update(train_critic(args.critic_warmup_steps, buffer))
        if ddp.rank == 0:
            print("object WM / critic warm-up:", {k: round(v, 4) for k, v in warm.items()}, flush=True)
    best = -float("inf")
    best_success, best_policy = -1.0, None
    since_collect, force_collect = args.collect_every, False
    for cycle in range(start, args.cycles):
        tic = time.perf_counter()
        sim_stats, fresh, wm_stats, control, anchor = {}, {}, {}, {}, {}
        if force_collect or since_collect >= args.collect_every:
            since_collect, force_collect = 0, False
            episodes, sim_stats = collect()
            fresh = {**wm_fresh(episodes), **critic_fresh(episodes)}
            if args.sim_anchor:
                anchor = sim_anchor(episodes)
            episodes.noise = episodes.log_prob = None  # only the anchor needs them
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
            wm_stats = {**train_wm(args.wm_steps), **train_critic(args.critic_steps, buffer)}
        else:
            t_sim = time.perf_counter()
        since_collect += 1
        t_wm = time.perf_counter()
        rank = {}
        if args.rank_every and cycle % args.rank_every == 0:
            g = n // args.rank_group
            rows = train_rows[torch.randint(len(train_rows), (g,), generator=rng)]
            delay = int(torch.randint(args.rank_max_delay + 1, (1,), generator=rng))
            rank = _all_mean(collector.ranking(noise_policy, imaginer, rows, args.rank_group, delay), ddp, device)
            tau = rank["rank/tau"]
            if math.isnan(tau) or tau < args.tau_min:
                control["imag_skipped"] = 1.0
                force_collect = True
            elif not math.isnan(tau):
                if tau < args.tau_low:
                    imag_iters = max(args.imag_iters_min, imag_iters // 2)
                elif tau > args.tau_high:
                    imag_iters = min(args.imag_iters_max, imag_iters * 2)
        t_rank = time.perf_counter()
        probe = buffer.starts(args.imag_starts)["cond"]
        with torch.no_grad():
            ref_mean, ref_log_std = noise_policy(probe)
        imag_stats, drift = [], 0.0
        for _ in range(0 if control.get("imag_skipped") else imag_iters):
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
            **_all_mean({**sim_stats, **fresh, **wm_stats, **imag, **anchor}, ddp, device), **rank,
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
