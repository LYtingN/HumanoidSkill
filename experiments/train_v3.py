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
2. object WM: ``wm_steps`` steps on windows of ``wm_history`` past and ``horizon`` future frames,
   the future keypoints simulated or (for ``phi_fraction`` of them) Phi's, labels always the
   simulated can; with ``events`` the event head (continue / fail / done per step, read off the WM's
   prediction) on the same batches, which then scores the imagination with probabilities;
3. ranking check (every ``rank_every`` cycles): candidates from identical sim states, each executed
   for one chunk (``execute_steps``, up to the next replan) in sim vs imagined over the same steps
   (``rank/tau``), and the ideal branch against the candidates S0 executed exactly
   (``rank/ideal_*``, ``rank/gap_*``); with ``rank_ideal`` each candidate is also executed by the
   simulated ideal executor E*, the ideal branch's ground truth (``rank/sim_*``: J* simulated vs
   imagined, ``rank/episode_gap``: the episode's J* - J of one decision); below ``tau_min`` the cycle skips imagination and the
   next one collects (the imagined ranking is no better than chance); a low tau halves the
   imagined iterations per sim rollout, a high tau doubles them, within
   [``imag_iters_min``, ``imag_iters_max``];
4. imagination: up to ``imag_iters`` GRPO updates, each on ``imag_starts`` recorded decisions x
   ``imag_group`` noises scored by ``hskill.executors.imaginer.Imaginer`` (chunk returns of the sim task
   potential, see ``can_pnp.task_spec``; no value beyond the chunk while the critic is redesigned) on the
   actual branch J, plus ``intent_weight`` x the ideal branch's excess max(J* - J, 0) (0 while S0
   stays frozen: an intent S0 cannot execute is worth what S0 achieves); candidates the object WM
   ensemble disagrees on by more than ``uncertainty_max`` get no advantage; each
   stopped at ``imag_target_kl``, and all once the policy moved ``cycle_kl`` from the cycle's
   start (the simulator has the last word).

With ``critic`` the model-based critic (``hskill.learners.critic``) learns the episode return from
the imagined window of ``critic_samples`` recorded decisions per collection (their executed noise),
is scored on them before training on them (``critic_fresh/*``), and the ranking check continues
the candidates' episodes to their end (``rank/critic_*``). ``--no-imagine`` turns the imagined
updates off: every cycle collects, the noise policy learns from the simulator only (sim GRPO) and
the world models and the critic are only fitted and checked.

With ``value`` the critic is instead the value of the simulated decision states: it learns from the
features of the real state at every recorded decision of the policy envs, with TD(``gae_lambda``)
targets (lambda-returns) instead of Monte-Carlo ones, and imagination scores a chunk's predicted
end with it (short model rollout + learned value). ``advantage gae`` replaces the simulator's GRPO
update (one normalized return per episode, shared by its decisions) with per-decision GAE
advantages on that value, normalized over the batch.

``curriculum`` (grasp-start phases from easy to hard, e.g. RELEASE LOWER PLACE_REACH TURN_WALK LIFT
CLOSE) is a reverse curriculum: collection groups start at the frontier phase (a
``curriculum_frontier`` share of them, the others on an easier phase already passed), and the
frontier moves one phase back once the noise policy's success rate there reaches
``curriculum_advance``; the ranking check starts at the frontier. Without it every start is a
``grasp_phase`` one.
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

import numpy as np
import torch
import torch.distributed as dist
import tyro
from can_pnp.task_spec import TaskTracker
from hskill.data.buffer import TIME_MAJOR, EpisodeBuffer
from hskill.executors.imagine import PhiStep
from hskill.executors.imaginer import Imaginer
from hskill.interaction.events import EventHead, event_loss
from hskill.interaction.model import ObjectWMCfg, ObjectWMEnsemble, ObjectWorldModel, object_loss, object_targets
from hskill.learners.critic import ModelCritic, critic_features
from hskill.learners.grpo import Distributed, gae_advantages, group_advantages, masked_group_advantages, update
from hskill.s0.body_world_model import DEFAULT_REPO, BodyWorldModel
from hskill.s0.kinematics import BodyKinematics
from hskill.s1.noise_policy import NoisePolicy, SteeredDiffusionSource
from hskill.sim.collect import SimCollector
from hskill.sim.rollouts import Rollouts
from hskill.utils import distributions

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
    compile_mode: str | None = None  # torch.compile mode of the diffusion policy (DINOv3 + DiT)
    # Diffusion sampling; None keeps the checkpoint's (replan every 8 control steps, 16 DDIM steps).
    execute_steps: int | None = None  # control steps per reference before replanning
    ddim_steps: int | None = None
    rgb_backend: Literal["mujoco_warp", "hyperlux"] = "mujoco_warp"
    render_size: tuple[int, int] = (320, 240)
    # Task (as train_dsrl).
    success_reward: float = 10.0
    dist_scales: tuple[float, ...] = (0.1, 0.5)  # m: the shaping's length scales
    dist_weights: tuple[float, ...] = (0.5, 0.5)
    success_hold_s: float = 0.2
    upright_cos: float = 0.866
    release_opening: float = 0.03
    horizon: int = 32  # imagined frames per decision: the whole reference
    # Object world model.
    wm_history: int = 10  # past frames of keypoints, openings and can before the predicted horizon
    wm_hidden: tuple[int, ...] = (512, 512, 512)
    wm_lr: float = 3e-4
    wm_batch: int = 4096
    wm_warmup_rollouts: int = 2
    wm_warmup_steps: int = 2000
    wm_steps: int = 200
    # Event head (hskill.interaction.events): the imagination scores the probabilities it reads off the
    # object WM's prediction (S_H, P_done) instead of thresholds on predicted states, once it has had
    # event_min_steps updates (trained on each WM batch, the WM's predictions as its input).
    events: bool = True
    event_hidden: tuple[int, ...] = (256, 256)
    event_lr: float = 3e-4
    event_min_steps: int = 500
    phi_fraction: float = 0.5
    phi_env_fraction: float = 1.0  # share of the collection envs Phi runs windows for (the Phi-body inputs)
    # Share of the exactly executed windows (right hand within well_tracked) given the ideal branch's
    # keypoints, so the object WM also learns that input (labels: the simulated can).
    wm_ideal_fraction: float = 0.5
    wm_output: Literal["delta", "rigid"] = "delta"  # object WM output, see hskill.interaction.model
    wm_ensemble: int = 3  # object WM members; their disagreement gates the imagined candidates
    uncertainty_max: float = 0.03  # m: final object position spread beyond which a candidate is not trusted
    buffer_rollouts: int = 8
    buffer_decay: float = 0.7  # sampling weight per rollout of age
    # Model-based critic (evaluation only: it does not score imagined candidates yet).
    critic: bool = False
    critic_samples: int = 2048  # recorded decisions per collection and GPU whose window the critic gets
    critic_pool: int = 32768  # (features, return) pairs kept per GPU, newest first
    critic_steps: int = 200
    critic_batch: int = 1024
    critic_lr: float = 3e-4
    critic_hidden: tuple[int, ...] = (512, 512, 512)
    critic_members: int = 3
    # The critic as the value of the simulated decision states, TD(gae_lambda) targets (see above).
    value: bool = False
    # The simulator's update: "grpo" (group-normalized episode returns) or "gae" (per decision, on
    # the value; GRPO until the critic's pool holds imag_critic_min_pool samples).
    advantage: Literal["grpo", "gae"] = "grpo"
    gae_lambda: float = 0.9
    # Weight of the critic in the imagined score (0: the chunk's outcome only; 1: the chunk end's
    # value by the critic), once its pool holds imag_critic_min_pool samples of the current object WM.
    imag_critic_weight: float = 1.0
    imag_critic_min_pool: int = 4096
    # Reverse curriculum: start phases from easy to hard; empty: grasp_phase only.
    curriculum: tuple[str, ...] = ()
    curriculum_advance: float = 0.5  # policy success rate at the frontier phase that moves it back one phase
    curriculum_frontier: float = 0.75  # share of the collection groups at the frontier, the rest on easier phases
    # Episode length (s) per curriculum phase; a collection (and the ranking check) runs the
    # frontier's. Empty: episode_s for every phase.
    curriculum_episode_s: tuple[float, ...] = ()
    # Imagined GRPO (off: no imagined updates, every cycle collects).
    imagine: bool = True
    imag_starts: int = 256
    imag_group: int = 16
    imag_iters: int = 8
    imag_lr: float | None = None  # the imagined updates' lr (smaller, noisier batches than the simulator's); None: lr
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
    # Ideal branch: the reference executed exactly, the whole body ("full": all five keypoints from
    # the reference) or only the arms on the actual body ("arms": pelvis and feet as executed).
    ideal_branch: Literal["full", "arms"] = "full"
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
    # Pair the ranking check's candidates: each also executed by the simulated ideal executor E*
    # (hskill.executors.ideal_body), the ideal branch's ground truth (rank/sim_*, rank/episode_gap);
    # rank/tau then ranks rank_group // 2 candidates per group.
    rank_ideal: bool = True
    ideal_kp_scale: float = 10.0  # E*'s joint stiffness x the actuators' (x3: 4 cm hand error, x30: 1 cm)
    # Share of the collection groups (of the noise policy's) executed by E* throughout: their windows
    # train the object WM on near-exact executions (the ideal branch's input, with its simulated
    # outcome), and sim_ideal/* tells how the policy does when executed exactly; nothing of them
    # trains the policy, the critic or counts for the curriculum.
    ideal_collect_fraction: float = 0.0625
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
    if args.rank_ideal and args.rank_group % 2:
        raise ValueError("--rank-ideal pairs the candidates: --rank-group must be even")
    collector = SimCollector(rollouts, phi, kin, args.horizon, args.well_tracked, args.wm_history, args.ideal_branch,
                             args.phi_env_fraction,
                             ideal_executor=(args.rank_ideal and bool(args.rank_every)) or args.ideal_collect_fraction > 0,
                             ideal_kp_scale=args.ideal_kp_scale)  # fmt: skip
    noise_policy = NoisePolicy(diffusion.n_tokens, diffusion.n_emb, diffusion.horizon, diffusion.action_dim,
                               hidden=args.hidden, init_log_std=args.init_log_std).to(device)  # fmt: skip
    optimizer = torch.optim.Adam(noise_policy.parameters(), lr=args.lr)
    wm_cfg = ObjectWMCfg(history=args.wm_history, horizon=args.horizon, hidden=args.wm_hidden, dt=env.step_dt,
                         output=args.wm_output)  # fmt: skip
    object_wm = (ObjectWMEnsemble(wm_cfg, args.wm_ensemble) if args.wm_ensemble > 1 else ObjectWorldModel(wm_cfg)).to(device)
    if args.compile:
        object_wm.compile_step()
    wm_optimizer = torch.optim.Adam(object_wm.parameters(), lr=args.wm_lr)
    event_head = EventHead(args.event_hidden).to(device)
    event_optimizer = torch.optim.Adam(event_head.parameters(), lr=args.event_lr)
    event_steps = 0
    rim = rollouts.geo.box_bottom_z + rollouts.geo.upper_box_size[2]
    decoder = SteeredDiffusionSource(diffusion, noise_policy, bf16=args.bf16)
    if (args.value or args.advantage == "gae") and not args.critic:
        raise ValueError("--value and --advantage gae need --critic")
    if args.advantage == "gae" and not args.value:
        raise ValueError("--advantage gae needs the critic as the value (--value)")
    critic = ModelCritic(args.critic_hidden, args.critic_members).to(device)
    imaginer = Imaginer(decoder, phi, kin, object_wm, rollouts.task, rollouts.geo, args.horizon, args.imag_batch,
                        task_tracker=TaskTracker, ideal=args.ideal_branch, critic=critic if args.critic else None)  # fmt: skip
    imag_cfg = dataclasses.replace(args, target_kl=args.imag_target_kl)
    buffer = EpisodeBuffer(args.buffer_rollouts, args.buffer_decay, args.horizon, args.wm_history)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    pool: list[tuple[torch.Tensor, torch.Tensor]] = []
    start, imag_iters = 0, args.imag_iters
    wm_warm_up = args.resume is None
    level = 0  # curriculum frontier: index into rollouts.starts.phases
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        noise_policy.load_state_dict(state["noise_policy"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:  # the optimizer state keeps the old run's lr
            group["lr"] = args.lr
        if state["args"].get("wm_output", "rigid") == args.wm_output:
            object_wm.load_state_dict(state["object_wm"])
            wm_optimizer.load_state_dict(state["wm_optimizer"])
        else:  # another output parameterization: the object WM starts over (warmed up below)
            wm_warm_up = True
            print(f"object WM: checkpoint {state['args'].get('wm_output', 'rigid')} != {args.wm_output}, trained afresh", flush=True)
        start, imag_iters = state["cycle"] + 1, state["imag_iters"]
        level = state.get("curriculum_level", 0)
        if "critic" in state:
            critic.load_state_dict(state["critic"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "event_head" in state:
            event_head.load_state_dict(state["event_head"])
            event_optimizer.load_state_dict(state["event_optimizer"])
            event_steps = state["event_steps"]
        args.cycle_kl = state.get("cycle_kl", args.cycle_kl)
    elif args.init_noise_checkpoint is not None:
        # Not its optimizer: Adam's second moments of the large sim batches would blow up the
        # steps on the (smaller, noisier) imagined batches.
        noise_policy.load_state_dict(torch.load(args.init_noise_checkpoint, map_location=device,
                                                weights_only=False)["noise_policy"])  # fmt: skip
    ddp.broadcast(noise_policy)
    ddp.broadcast(object_wm)
    ddp.broadcast(critic)
    log = dists_log = None
    if ddp.rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "args.json").write_text(json.dumps(asdict(args), default=str, indent=2))
        log = (args.out_dir / "metrics.jsonl").open("a")
        # Per-collection histograms (success / S1 / S0 at the frontier; tools/plot_distributions.py).
        dists_log = (args.out_dir / "dists.jsonl").open("a")
        print(f"v3: {ddp.world} GPU x {args.num_envs} sim envs, imagination {args.imag_starts} x {args.imag_group}",
              flush=True)  # fmt: skip
    rng = torch.Generator().manual_seed(args.seed + 7919 * ddp.rank)
    train_rows = torch.tensor(rollouts.train_rows)
    phases = rollouts.starts.phases
    if args.curriculum_episode_s and len(args.curriculum_episode_s) != len(phases):
        raise ValueError(f"curriculum_episode_s needs one length per phase {phases}")
    phase_rows = [train_rows[rollouts.starts.phase_ids[train_rows] == i] for i in range(len(phases))]
    n = env.num_envs

    if n % args.sim_group:
        raise ValueError(f"{n} envs do not split into groups of {args.sim_group}")
    groups = n // args.sim_group

    def sample_rows(count: int, phase: torch.Tensor) -> torch.Tensor:
        """One random training start of each given phase (count,)."""
        return torch.stack([phase_rows[i][int(torch.randint(len(phase_rows[i]), (1,), generator=rng))]
                            for i in phase.tolist()]) if count else torch.zeros(0, dtype=torch.long)  # fmt: skip

    def collect():
        """Episodes, stats and each env's start phase (N,)."""
        phase = torch.full((groups,), level)
        if level > 0:  # the groups off the frontier replay an easier phase
            easy = torch.rand(groups, generator=rng) >= args.curriculum_frontier
            phase[easy] = torch.randint(level, (int(easy.sum()),), generator=rng)
        rows = sample_rows(groups, phase).repeat_interleave(args.sim_group)
        base = torch.rand(groups, generator=rng) < args.base_fraction
        ideal = ~base & (torch.rand(groups, generator=rng) < args.ideal_collect_fraction)
        base, ideal = base.repeat_interleave(args.sim_group), ideal.repeat_interleave(args.sim_group)
        episodes, stats = collector.collect(noise_policy, rows, base.to(device), episode_seconds(), ideal.to(device))
        return episodes, stats, phase.repeat_interleave(args.sim_group).to(device)

    def episode_seconds() -> float:
        """Episode length of the current frontier phase."""
        return args.curriculum_episode_s[level] if args.curriculum_episode_s else args.episode_s

    def phase_success(episodes, env_phase: torch.Tensor) -> dict:
        """The noise policy's success rate per curriculum phase (over all ranks)."""
        policy = episodes.policy_envs()
        counts = torch.stack([torch.stack([(episodes.placed & policy & (env_phase == i)).sum(),
                                           (policy & (env_phase == i)).sum()]).float()
                              for i in range(len(phases))])  # fmt: skip
        counts = ddp.sum(counts)
        return {f"curriculum/success_{name}": float(c[0] / c[1]) for name, c in zip(phases, counts) if c[1] > 0}

    def sim_anchor(episodes, gae: torch.Tensor | None = None) -> dict:
        """One update on the policy envs' episodes: GRPO on the groups' episode returns (DSRL's
        update), or the per-decision ``gae`` advantages (D, N) normalized over all ranks' decisions."""
        mask = (episodes.decision_live & episodes.policy_envs()[None]).float()
        if gae is None:
            adv, informative = group_advantages(episodes.final_potential, args.sim_group)
            info = {"anchor/informative_groups": float(informative.mean())}
        else:
            s1, s2, count = ddp.sum(torch.stack([(gae * mask).sum(), (gae.square() * mask).sum(), mask.sum()]))
            mean = s1 / count.clamp_min(1)
            std = (s2 / count.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
            adv = (gae - mean) / (std + 1e-6) * mask
            info = {"anchor/gae_mean": float(mean), "anchor/gae_std": float(std)}
        batch = {"cond": episodes.cond, "noise": episodes.noise, "log_prob": episodes.log_prob, "mask": mask}
        stats = update(noise_policy, optimizer, batch, adv, args, ddp)
        return info | {"anchor/gae": float(gae is not None)} | {f"anchor/{k}": float(v) for k, v in stats.items()}

    def wm_batch_loss(source: EpisodeBuffer, size: int, phi_fraction: float, ideal_fraction: float = 0.0,
                      exact_only: bool = False, phi_only: bool = False, windows: bool = False):  # fmt: skip
        b = source.windows(size, phi_fraction, ideal_fraction, args.well_tracked, exact_only, phi_only)
        pred, logits = object_wm.rollout(b["hist_kp"], b["hist_width"], b["hist_obj"], b["fut_kp"], b["fut_cmd"])
        loss, m = object_loss(pred, logits, object_targets(b["obj"], env.step_dt), b["held"], b["mask"])
        return (loss, m, pred, b) if windows else (loss, m)

    def wm_members_loss(size: int):
        """Every ensemble member on its own batch (one stacked rollout): the members' spread then
        reflects the data's coverage."""
        if not isinstance(object_wm, ObjectWMEnsemble):
            return wm_batch_loss(buffer, size, args.phi_fraction, args.wm_ideal_fraction, windows=True)
        E = object_wm.num_members
        b = buffer.windows(E * size, args.phi_fraction, args.wm_ideal_fraction, args.well_tracked)
        H = b["fut_cmd"].shape[0]
        # The windows come grouped by episode: shuffle before splitting them among the members.
        perm = torch.randperm(E * size, device=device)
        b = {k: v[:, perm] if k in TIME_MAJOR else v[perm] for k, v in b.items()}
        inputs = [b[k].view(b[k].shape[0], E, size, *b[k].shape[2:])
                  for k in ("hist_kp", "hist_width", "hist_obj", "fut_kp", "fut_cmd")]  # fmt: skip
        objs, logits = object_wm.rollout_members(*inputs)
        pred = objs.transpose(0, 1).reshape(H + 1, E * size, -1)
        loss, m = object_loss(pred, logits.transpose(0, 1).reshape(H, E * size), object_targets(b["obj"], env.step_dt),
                              b["held"], b["mask"])  # fmt: skip
        return loss, {**m, "ideal_share": b["ideal"].float().mean()}, pred, b

    def train_events(pred: torch.Tensor, b: dict) -> dict:
        """One event-head step on a WM batch: the WM's (detached) predictions in, the episodes' ends as labels."""
        nonlocal event_steps
        event_head.train()
        logits = event_head(*EventHead.window_inputs(pred.detach(), b["hist_kp"], b["fut_kp"], b["fut_cmd"]), b["box"], rim)
        loss, m = event_loss(logits, b["fail"], b["done"], b["mask"])
        event_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        ddp.average_grads(event_head)
        grad = torch.nn.utils.clip_grad_norm_(event_head.parameters(), 10.0)
        if not bool(ddp.sum((~(loss.isfinite() & grad.isfinite())).float().reshape(1)) > 0):
            event_optimizer.step()
            event_steps += 1
        event_head.eval()
        return m

    @contextlib.contextmanager
    def tf32():
        """TF32 matrix products while fitting the object WM (its fp32 matmuls are most of that
        time); imagination and the policies keep full fp32."""
        # Legacy flag, as make_env sets it: mixing in fp32_precision breaks torch.compile's max-autotune.
        matmul = torch.backends.cuda.matmul
        prev, matmul.allow_tf32 = matmul.allow_tf32, True
        try:
            yield
        finally:
            matmul.allow_tf32 = prev

    @tf32()
    def train_wm(steps: int) -> dict:
        object_wm.train()
        metrics = []
        events = []
        for _ in range(steps):
            loss, m, pred, b = wm_members_loss(args.wm_batch)
            if args.events:
                events.append(train_events(pred, b))
            wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ddp.average_grads(object_wm)
            grad = torch.nn.utils.clip_grad_norm_(object_wm.parameters(), 10.0)
            if bool(ddp.sum((~(loss.isfinite() & grad.isfinite())).float().reshape(1)) > 0):
                continue  # a non-finite step on any rank skips it everywhere
            wm_optimizer.step()
            metrics.append({"loss": loss.detach(), **m})
        object_wm.eval()
        if args.events:  # the imagination scores with the head once it has trained enough
            imaginer.event_head = event_head if event_steps >= args.event_min_steps else None
        return ({f"wm/{k}": v for k, v in _mean(metrics[-20:]).items()}
                | {f"event/{k}": v for k, v in _mean(events[-20:]).items()}
                | ({"event/steps": float(event_steps), "event/scoring": float(imaginer.event_head is not None)}
                   if args.events else {}))  # fmt: skip

    @torch.no_grad()
    def wm_fresh(episodes) -> dict:
        fresh = EpisodeBuffer(1, 1.0, args.horizon, args.wm_history)
        fresh.add(episodes)
        if not len(fresh.valid[0]):
            return {}
        out = {}
        # The ideal branch's input where the reference was executed exactly (labels then valid).
        loss, m = wm_batch_loss(fresh, args.wm_batch, 0.0, 1.0, exact_only=True)
        out.update({f"wm_fresh/ideal_body_{k}": float(v) for k, v in {"loss": loss, **m}.items()
                    if k in ("loss", "pos_err_cm_final", "pos_err_cm_mean")})  # fmt: skip
        for name, phi in (("sim_body", 0.0), ("phi_body", 1.0)):
            loss, m = wm_batch_loss(fresh, args.wm_batch, phi, phi_only=phi > 0)
            out.update({f"wm_fresh/{name}_{k}": float(v) for k, v in {"loss": loss, **m}.items()
                        if k in ("loss", "pos_err_cm_final", "pos_err_cm_mean", "held_acc")})  # fmt: skip
        return out

    @torch.no_grad()
    def critic_samples(episodes) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Features of the imagined windows of recorded decisions (policy envs, their executed noise;
        the seconds of their episode left) and their episodes' returns."""
        fresh = EpisodeBuffer(1, 1.0, args.horizon, args.wm_history)
        fresh.add(episodes)
        if not len(fresh.valid_policy[0]):
            return None
        st = fresh.starts(args.critic_samples, policy_only=True)
        out = imaginer.chunk_return(st, st["noise"])
        feat = critic_features(body, out["s_end"], out["kp_end"], out["obj_end"], out["held_logit_end"], st["goal"],
                               (1.0 - st["time"]) / st["step_frac"] * env.step_dt)  # fmt: skip
        keep = feat.isfinite().all(-1) & st["final"].isfinite()
        return feat[keep], st["final"][keep]

    @torch.no_grad()
    def decision_features(episodes) -> torch.Tensor:
        """(D, N, FEAT_DIM) the critic's features of the simulated state at each decision (the
        measured held flag as a saturated logit, the seconds of the episode left)."""
        D, n = episodes.s0.shape[:2]
        t = torch.arange(D, device=device) * episodes.execute_steps
        t_left = ((episodes.kp.shape[0] - 1 - t) * env.step_dt).float().repeat_interleave(n)
        held = torch.where(episodes.held[t], 10.0, -10.0)
        feat = critic_features(body, episodes.s0.flatten(0, 1), episodes.kp[t].flatten(0, 1), episodes.obj[t, ..., :13].flatten(0, 1),
                               held.flatten(), episodes.goal.repeat(D, 1), t_left)  # fmt: skip
        return feat.view(D, n, -1)

    @torch.no_grad()
    def value_targets(episodes) -> tuple[torch.Tensor, ...]:
        """The policy envs' live decisions (D, N), their features, GAE advantages and lambda-returns
        on the critic as the value."""
        feat = decision_features(episodes)
        live = episodes.decision_live & episodes.policy_envs()[None] & feat.isfinite().all(-1)
        values = critic(feat.flatten(0, 1)).mean(0).view(live.shape)
        adv, ret = gae_advantages(values, live, episodes.final_potential, args.gae_lambda)
        return live, feat, adv, ret

    @torch.no_grad()
    def critic_fresh(feat: torch.Tensor, target: torch.Tensor) -> dict:
        """The critic on decisions it has not been trained on."""
        members = critic(feat)
        q = members.mean(0)
        res = q - target
        return {
            "critic_fresh/mae": float(res.abs().mean()),
            "critic_fresh/explained_var": float(1 - res.var() / target.var().clamp_min(1e-8)),
            "critic_fresh/corr": float(torch.corrcoef(torch.stack([q, target]))[0, 1]),
            "critic_fresh/member_std": float(members.std(0).mean()),
            "critic_fresh/target_mean": float(target.mean()), "critic_fresh/pred_mean": float(q.mean()),
        }  # fmt: skip

    def train_critic(steps: int) -> dict:
        if not pool:
            return {}
        feat = torch.cat([f for f, _ in pool])
        target = torch.cat([t for _, t in pool])
        critic.train()
        metrics = []
        for _ in range(steps):
            idx = torch.randint(len(target), (args.critic_batch,), device=device)
            members = critic(feat[idx])
            loss = (members - target[idx]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ddp.average_grads(critic)
            grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            if bool(ddp.sum((~(loss.isfinite() & grad.isfinite())).float().reshape(1)) > 0):
                continue
            critic_optimizer.step()
            with torch.no_grad():
                res = members.mean(0) - target[idx]
                metrics.append({"loss": loss.detach(), "explained_var": 1 - res.var() / target[idx].var().clamp_min(1e-8)})
        critic.eval()
        return {f"critic/{k}": v for k, v in _mean(metrics[-20:]).items()} | {"critic/pool": float(len(target))}

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
        # Candidates whose chunk returns differ by less than 0.1% of the start's shaping count as ties
        # (no ranking signal).
        min_std = 1e-3 * out["potential0"].view(-1, args.imag_group)[:, :1]
        adv, informative = masked_group_advantages(torch.nan_to_num(score), args.imag_group, keep, min_std)
        batch = {"cond": rep["cond"][None], "noise": w[None], "log_prob": log_prob[None], "mask": keep.float()[None]}
        for group in optimizer.param_groups:
            group["lr"] = args.imag_lr or args.lr
        stats = update(noise_policy, optimizer, batch, adv, imag_cfg, ddp)
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        trusted_ideal = out["uncertainty_ideal"] <= args.uncertainty_max
        return {
            # J* diagnostics on the candidates the members agree on for the ideal branch only.
            "imag/return_ideal": float(ideal[trusted_ideal].mean()), "imag/gap": float(gap[trusted_ideal].mean()),
            "imag/ideal_trusted_share": float(trusted_ideal.float().mean()),
            "imag/closable_share": float(((gap > 0.1) & trusted_ideal).float().mean()),
            "imag/uncertainty_cm": float(100 * out["uncertainty"].mean()),
            "imag/uncertainty_ideal_cm": float(100 * out["uncertainty_ideal"].mean()),
            "imag/kept_share": float(keep.float().mean()),
            "imag/return": float(returns.mean()), "imag/placed": float(out["placed"].float().mean()),
            "imag/dropped": float(out["dropped"].float().mean()), "imag/final_dist": float(out["final_dist"].mean()),
            "imag/fall_prob": float(out["fall_prob"].mean()), "imag/informative_groups": float(informative.mean()),
            "imag/value": float(out["value"].nanmean()),
            # How much the candidates of a start differ by the chunk outcome.
            "imag/chunk_return_group_std": float(out["chunk_returns"].view(-1, args.imag_group).std(1).nanmean()),
            **{f"grpo/{k}": float(v) for k, v in stats.items()},
        }  # fmt: skip

    # The buffer is not checkpointed: fill it first (and warm the object WM up on a new run).
    for _ in range(args.wm_warmup_rollouts):
        episodes, _, _ = collect()
        episodes.noise = episodes.log_prob = None
        if not args.imagine:
            episodes.cond = episodes.obs0 = None
        buffer.add(episodes)
    if wm_warm_up:
        warm = train_wm(args.wm_warmup_steps)
        if ddp.rank == 0:
            print("object WM warm-up:", {k: round(v, 4) for k, v in warm.items()}, flush=True)
    best = -float("inf")
    best_success, best_policy = -1.0, None
    since_collect, force_collect = args.collect_every, False
    for cycle in range(start, args.cycles):
        tic = time.perf_counter()
        sim_stats, fresh, wm_stats, control, anchor, crit = {}, {}, {}, {}, {}, {}
        if force_collect or since_collect >= args.collect_every or not args.imagine:
            since_collect, force_collect = 0, False
            episodes, sim_stats, env_phase = collect()
            sim_stats |= phase_success(episodes, env_phase)
            counts = distributions.episode_histograms(episodes, env_phase == level)
            counts = {k: ddp.sum(v) for k, v in counts.items()}
            if dists_log is not None:
                dists_log.write(json.dumps(distributions.record(cycle, level, counts)) + "\n")
                dists_log.flush()
                # Rank 0's frontier episodes: where each layer puts the can (tools/plot_layers.py).
                geo = rollouts.geo
                points = distributions.layer_points(episodes, env_phase == level, geo.box_bottom_z + geo.upper_box_size[2],
                                                    rollouts.env.step_dt)  # fmt: skip
                (args.out_dir / "layers").mkdir(exist_ok=True)
                np.savez_compressed(
                    args.out_dir / "layers" / f"{cycle:06d}.npz", cycle=cycle, phase=phases[level],
                    episode_s=episode_seconds(), outcomes=counts["success"].cpu().numpy(),
                    **{k: v.cpu().numpy().astype(np.float16) for k, v in points.items()},
                )  # fmt: skip
            fresh = wm_fresh(episodes)
            # The value is trusted for the GAE update once trained on imag_critic_min_pool samples.
            value_ready = sum(len(t) for _, t in pool) >= args.imag_critic_min_pool
            gae = None
            if args.critic and args.value:  # real decision states, lambda-return targets
                live, feat, gae, ret = value_targets(episodes)
                samples, mc = (feat[live], ret[live]), episodes.final_potential[None].expand_as(live)[live]
                if not len(mc):
                    samples = None
            else:
                samples = critic_samples(episodes) if args.critic else None
                mc = None if samples is None else samples[1]
            if samples is not None:
                if pool:  # scored against the episodes' returns before it trains on them
                    crit = critic_fresh(samples[0], mc)
                pick = torch.randperm(len(mc), device=device)[: args.critic_samples]
                pool.insert(0, (samples[0][pick], samples[1][pick]))
                while sum(len(t) for _, t in pool) - len(pool[-1][1]) >= args.critic_pool:
                    pool.pop()
            if args.sim_anchor:
                anchor = sim_anchor(episodes, gae if args.advantage == "gae" and value_ready else None)
            episodes.noise = episodes.log_prob = None  # only the anchor needs them
            if not args.imagine:  # imagination starts only (~3 GB per 2048-env rollout)
                episodes.cond = episodes.obs0 = None
            buffer.add(episodes)
            del episodes
            curriculum = {k: sim_stats.pop(k) for k in list(sim_stats) if k.startswith("curriculum/")}
            sim_stats = _all_mean(sim_stats, ddp, device) | curriculum
            frontier = curriculum.get(f"curriculum/success_{phases[level]}", float("nan"))
            success = frontier if args.curriculum else sim_stats.get("sim/success_rate", float("nan"))
            control["curriculum/level"], control["curriculum/frontier_success"] = float(level), frontier
            control["curriculum/episode_s"] = episode_seconds()
            if args.curriculum and frontier >= args.curriculum_advance and level < len(phases) - 1:
                level += 1
                best_success, best_policy = -1.0, None  # a harder frontier: no rollback to the easier one's best
                control["curriculum/advanced"] = 1.0
                print(f"curriculum: frontier {phases[level - 1]} at {frontier:.2f} -> {phases[level]}", flush=True)
            elif success > best_success:
                best_success = success
                best_policy = copy.deepcopy((noise_policy.state_dict(), optimizer.state_dict()))
            elif args.rollback_margin > 0 and best_success - success > args.rollback_margin:
                noise_policy.load_state_dict(best_policy[0])
                optimizer.load_state_dict(best_policy[1])
                args.cycle_kl *= 0.5
                control["rollback"] = 1.0
            t_sim = time.perf_counter()
            wm_stats = train_wm(args.wm_steps)
            if args.critic:
                crit |= train_critic(args.critic_steps)
                ready = sum(len(t) for _, t in pool) >= args.imag_critic_min_pool
                imaginer.critic_weight = args.imag_critic_weight if ready else 0.0
                crit["imag/critic_weight"] = imaginer.critic_weight
        else:
            t_sim = time.perf_counter()
        since_collect += 1
        t_wm = time.perf_counter()
        rank = {}
        if args.rank_every and cycle % args.rank_every == 0:
            g = n // args.rank_group
            rows = sample_rows(g, torch.full((g,), level))
            delay = int(torch.randint(args.rank_max_delay + 1, (1,), generator=rng))
            rank = collector.ranking(noise_policy, imaginer, rows, args.rank_group, delay, critic if args.critic else None,
                                     episode_seconds(), args.uncertainty_max, ideal_pairs=args.rank_ideal)  # fmt: skip
            rank = _all_mean(rank, ddp, device)
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
        if args.imagine:
            probe = buffer.starts(args.imag_starts)["cond"]
            with torch.no_grad():
                ref_mean, ref_log_std = noise_policy(probe)
        imag_stats, drift = [], 0.0
        for _ in range(0 if control.get("imag_skipped") or not args.imagine else imag_iters):
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
            **_all_mean({**sim_stats, **fresh, **wm_stats, **imag, **anchor, **crit}, ddp, device), **rank,
        }  # fmt: skip
        if ddp.rank == 0:
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            state = {
                "noise_policy": noise_policy.state_dict(), "optimizer": optimizer.state_dict(),
                "object_wm": object_wm.state_dict(), "wm_optimizer": wm_optimizer.state_dict(),
                "critic": critic.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                "event_head": event_head.state_dict(), "event_optimizer": event_optimizer.state_dict(),
                "event_steps": event_steps, "cycle": cycle, "imag_iters": imag_iters, "cycle_kl": args.cycle_kl, "args": asdict(args),
                "curriculum_level": level,
                "diffusion_checkpoint": str(args.diffusion_checkpoint),
                "tracker_checkpoint": str(args.tracker_checkpoint),
            }  # fmt: skip
            params = list(state["noise_policy"].values()) + list(state["object_wm"].values())
            if not all(bool(v.isfinite().all()) for v in params if v.is_floating_point()):
                raise RuntimeError(f"Non-finite parameters at cycle {cycle}; last.pt kept")
            if cycle % args.save_every == 0 or cycle == args.cycles - 1:
                torch.save(state, args.out_dir / "last.pt")
            # With a curriculum, a harder frontier beats any score on an easier one.
            # The level the episodes were collected at (before an advance in this cycle).
            score = (record.get("curriculum/level", level) + record.get("curriculum/frontier_success", 0.0) if args.curriculum
                     else record.get("sim/return", -float("inf")))  # fmt: skip
            if score > best:
                best = score
                torch.save(state, args.out_dir / "best.pt")
    if ddp.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
