"""DSRL + GRPO post-training of the can pick-and-place diffusion policy.

    # one GPU
    uv run --extra diffusion can-pnp-dsrl --tracker-checkpoint <supermimic>.pt \\
        --diffusion-checkpoint <Humanoid-Diffusion run>/best.pt --out-dir logs/dsrl/run1
    # eight GPUs (scripts/train_dsrl_8gpu.sh)
    torchrun --standalone --nproc-per-node 8 -m can_pnp.scripts.train_dsrl ...

Policy: :class:`can_pnp.dsrl.NoisePolicy` chooses the DDIM initial noise of the frozen diffusion
policy at every replan (8 control steps); the SuperMimic tracker executes the reference.

GRPO: every iteration each env group restarts from one recorded grasp start (all envs of a
group from the same state), rolls out ``episode_s`` seconds and gets one episodic return:
``success_reward`` once the task is done (the can rests released and upright in the box for
``success_hold_s``), otherwise ``exp(-d / dist_scale)`` of the can's distance ``d`` to the
placement goal at the end of the episode, 0 if it ended with a fall or a dropped can. The return
is a function of the final state only (a potential-based shaping summed over the episode
telescopes to it), so neither lingering near the goal nor a pass close to it before backing off
or crashing is worth anything; the distance term keeps returns within a group distinct before
any episode succeeds. Advantages are the returns normalized within each group; every noise
decision of an episode shares it. The update is PPO-clip on the noise log-likelihood, plus
``kl_coef`` x KL to N(0, I), which keeps the noise near the base policy's distribution.
Groups whose returns are all equal carry no signal and get zero advantage.

Diffusion replans and head-camera rendering dominate the rollout time. Replans run under bf16
autocast and torch.compile; the camera renders at ``render_size`` and only on the two control
steps whose images feed the next replan (``obs_horizon`` = 2).
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import tyro
from mjlab.utils.nan_guard import NanGuard

from can_pnp.config import CanPnpSceneCfg, SceneGeometryCfg
from can_pnp.dsrl import NoisePolicy, SteeredDiffusionSource
from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.grasp_start import GraspStarts, apply_grasp_start
from can_pnp.layout import layout_term
from can_pnp.pipeline import SuperMimicController, load_diffusion_policy
from can_pnp.supermimic import load_tracking_policy

# Held-out episodes of the diffusion policy's training split (dataset_report.json, seed 42).
VAL_EPISODES = (3, 46, 50, 54, 79, 120, 135, 169, 171, 204, 221, 245)


@dataclass
class Args:
    tracker_checkpoint: Path
    diffusion_checkpoint: Path
    diffusion_repo: Path = Path("../../Diffusion/Humanoid-Diffusion")
    out_dir: Path = Path("logs/dsrl/run")
    resume: Path | None = None
    # Rollouts (per GPU): groups x group_size envs, each group from one grasp start.
    groups: int = 32
    group_size: int = 8
    episode_s: float = 14.0
    grasp_phase: Literal["REACH", "CLOSE"] = "CLOSE"
    can_mass: float = 0.1
    # Returns: success_reward, else exp(-final can-goal distance / dist_scale) (0 after a fall / drop).
    success_reward: float = 10.0
    dist_scale: float = 0.1  # m
    # Done: can in the box, upright (axis within acos(upright_cos) of vertical) and the right
    # gripper commanded open (> release_opening), all for success_hold_s.
    success_hold_s: float = 0.2
    upright_cos: float = 0.866
    release_opening: float = 0.03  # m; the recorded release opens to 0.065, the grasp closes to 0
    bf16: bool = True  # diffusion (DINOv3 + DiT) inference under bf16 autocast, as it was trained
    compile: bool = True  # torch.compile DINOv3 and the DiT (~25% faster replans, ~1 min warm-up)
    rgb_backend: Literal["mujoco_warp", "hyperlux"] = "mujoco_warp"  # head-camera renderer
    # Head-camera render size (same 4:3 view). The policy sees it area-resized to 224 x 224; at
    # 320 x 240 that image is within 0.8/255 of the 640 x 480 render and renders 3x faster.
    render_size: tuple[int, int] = (320, 240)
    # GRPO / PPO.
    iterations: int = 2000
    # The noise has horizon x action_dim = 1184 dims and its log-likelihood sums over them, so a
    # step moves the policy ~sqrt(1184) times more than in a typical PPO action space.
    lr: float = 2e-5
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    kl_coef: float = 1e-3
    target_kl: float = 0.05  # stop the update once an epoch's approx KL (summed over dims) passes it
    max_grad_norm: float = 1.0
    hidden: int = 512
    init_log_std: float = 0.0
    # Bookkeeping.
    eval_every: int = 25  # deterministic (mean noise) rollouts on VAL_EPISODES; 0 disables
    save_every: int = 25
    seed: int = 0


class Distributed:
    def __init__(self):
        self.enabled = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        local = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = f"cuda:{local}"
        torch.cuda.set_device(self.device)
        if self.enabled:
            dist.init_process_group("nccl", device_id=torch.device(self.device))

    def sum(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(x)
        return x

    def average_grads(self, module: torch.nn.Module) -> None:
        if not self.enabled:
            return
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        flat = torch.cat([g.flatten() for g in grads])
        dist.all_reduce(flat)
        flat /= self.world
        offset = 0
        for g in grads:
            g.copy_(flat[offset : offset + g.numel()].view_as(g))
            offset += g.numel()

    def broadcast(self, module: torch.nn.Module) -> None:
        if self.enabled:
            for p in module.parameters():
                dist.broadcast(p.data, 0)


class Rollouts:
    """Batched grasp-start episodes of the steered diffusion policy."""

    def __init__(self, args: Args, ddp: Distributed):
        self.args, self.device = args, ddp.device
        n = args.groups * args.group_size
        self.env = make_env(
            CanPnpEnvCfgArgs(
                # Episodes are timed here, not by the env: its time-out would reset envs mid-rollout.
                num_envs=n, randomize=False, camera=True, rgb_backend=args.rgb_backend, episode_length_s=1e6,
                seed=args.seed + ddp.rank,
                scene=CanPnpSceneCfg(
                    geometry=SceneGeometryCfg(can_mass=args.can_mass),
                    camera=dataclasses.replace(CanPnpSceneCfg().camera, width=args.render_size[0], height=args.render_size[1]),
                ),
            ),
            device=self.device,
        )  # fmt: skip
        self.diffusion, _ = load_diffusion_policy(args.diffusion_checkpoint, args.diffusion_repo, self.device)
        self.diffusion.requires_grad_(False)
        if args.compile:
            self.diffusion.compile_for_training()
        self.tracker = load_tracking_policy(args.tracker_checkpoint, device=self.device)
        self.starts = GraspStarts(phase=args.grasp_phase)
        val = set(VAL_EPISODES)
        self.train_rows = [i for i, e in enumerate(self.starts.episode_ids) if e not in val]
        self.val_rows = [i for i, e in enumerate(self.starts.episode_ids) if e in val]
        self.sensors = self.env.sim._sensor_context
        self.geo = self.env.cfg.events["reset_layout"].params["scene_cfg"].geometry
        self.env.reset()

    def _render(self, on: bool) -> None:
        # mjlab renders every env step in sim.sense(); without the context it skips camera work.
        self.env.sim._sensor_context = self.sensors if on else None

    @torch.no_grad()
    def run(self, noise_policy: NoisePolicy, rows: torch.Tensor, mode: str) -> dict:
        """One episode per env from grasp start ``rows`` (indices into ``self.starts``)."""
        env, args, device = self.env, self.args, self.device
        n = env.num_envs
        source = SteeredDiffusionSource(self.diffusion, noise_policy, mode, bf16=args.bf16)
        controller = SuperMimicController(env, source, self.tracker)
        self._render(True)
        # Rollouts follow each other without env resets: restart the episode clock and the action
        # history (the tracker observes its last actions) before writing the grasp starts.
        env.episode_length_buf.zero_()
        env.action_manager.reset()
        # A world whose physics diverged in an earlier rollout keeps NaNs (e.g. in the solver warm
        # start) that writing a start state does not clear: reset it fully first.
        stale = NanGuard.detect_nans(env.sim.data)
        if bool(stale.any()):
            env.reset(env_ids=stale.nonzero().flatten())
        apply_grasp_start(env, torch.arange(n, device=device), self.starts, rows)
        obs = env.obs_buf
        reset = torch.ones(n, dtype=torch.bool, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        lifted, entered, placed, fallen, dropped, diverged = (torch.zeros(n, dtype=torch.bool, device=device) for _ in range(6))
        hold = torch.zeros(n, dtype=torch.long, device=device)
        hold_steps = max(1, round(args.success_hold_s / env.step_dt))
        decisions_alive = []
        can = env.scene["can"]
        box = layout_term(env).box_center_w
        goal = layout_term(env).goal_pos_w
        final_dist = (can.data.root_link_pos_w - goal).norm(dim=-1)
        min_dist = final_dist.clone()
        envs = torch.arange(n, device=device)
        inner = torch.tensor(self.geo.upper_box_size[:2], device=device) / 2 - self.geo.box_wall_thickness
        k = source.execute_steps
        for step in range(round(args.episode_s / env.step_dt)):
            if step % k == 0:
                decisions_alive.append(alive.clone())
            action = controller.act(obs, reset)
            reset = torch.zeros_like(reset)
            released = controller.plan_gripper[envs, controller.step_in_chunk - 1, 1] > args.release_opening
            # Images feeding the next replan: after control steps k-2 and k-1 of the chunk.
            self._render(step % k >= k - 2)
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(truncated.any()):
                raise RuntimeError("Env time-out inside a rollout")
            # Terminated envs are already reset: their causes come from the termination terms,
            # and only the others' scene state belongs to this episode.
            terms = env.termination_manager
            fell = alive & terms.get_term("fallen")
            fallen |= fell
            dropped |= alive & terms.get_term("can_dropped") & ~fell
            # Physics blow-up (NaN / Inf state): a failed episode like a fall; its state is unusable.
            nan = alive & NanGuard.detect_nans(env.sim.data)
            diverged |= nan
            live = alive & ~terminated & ~nan
            can_z = can.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            rel = (can.data.root_link_pos_w[:, :2] - box[:, :2]).abs()
            in_box = (rel < inner).all(-1) & (can_z < self.geo.box_bottom_z + self.geo.can_height)
            q = can.data.root_link_quat_w  # (w, x, y, z); can axis z component: 1 - 2(x^2 + y^2)
            upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square()) > args.upright_cos
            lifted |= live & (can_z > self.geo.table_height + self.geo.can_height / 2 + 0.05)
            entered |= live & in_box
            hold = torch.where(live & in_box & upright & released, hold + 1, 0)
            placed |= hold >= hold_steps
            dist = (can.data.root_link_pos_w - goal).norm(dim=-1)
            final_dist = torch.where(live, dist, final_dist)
            min_dist = torch.where(live, torch.minimum(min_dist, dist), min_dist)
            # An episode ends at success, at a fall or with the can dropped.
            alive &= ~(placed | terminated | nan)
        self._render(True)
        shaped = torch.where(fallen | dropped | diverged, 0.0, torch.exp(-final_dist / args.dist_scale))
        returns = torch.where(placed, args.success_reward, shaped).float()
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
            "returns": returns, "lifted": lifted, "entered": entered, "placed": placed, "fallen": fallen,
            "dropped": dropped, "diverged": diverged, "final_dist": final_dist, "min_dist": min_dist,
        }  # fmt: skip


def group_advantages(returns: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns normalized within each group of consecutive envs; zero for constant groups."""
    r = returns.view(-1, group_size)
    std = r.std(dim=1, keepdim=True)
    informative = std > 1e-6
    adv = torch.where(informative, (r - r.mean(dim=1, keepdim=True)) / (std + 1e-6), torch.zeros_like(r))
    return adv.flatten(), informative.flatten().float()  # (envs,), (groups,)


def update(noise_policy: NoisePolicy, optimizer: torch.optim.Optimizer, batch: dict, adv: torch.Tensor,
           args: Args, ddp: Distributed) -> dict:  # fmt: skip
    d, n = batch["mask"].shape
    cond = batch["cond"].flatten(0, 1)
    noise, old = batch["noise"].flatten(0, 1), batch["log_prob"].flatten(0, 1)
    mask = batch["mask"].flatten()
    adv = adv[None].expand(d, n).flatten()
    total = ddp.sum(mask.sum().clone()).clamp_min(1.0)
    stats = {"loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "kl_base": 0.0, "updates": 0, "skipped_steps": 0}
    stop = False
    for _ in range(args.epochs):
        order = torch.randperm(len(mask), device=mask.device)
        epoch_kl = torch.zeros(2, device=mask.device)  # summed KL, live decisions seen
        for chunk in order.chunk(args.minibatches):
            m = mask[chunk]
            mean, log_std = noise_policy(cond[chunk])
            log_prob = NoisePolicy.log_prob(noise[chunk], mean, log_std)
            # Masked decisions get ratio 1, so no exp of their (meaningless) log-ratio enters backward.
            log_ratio = torch.where(m > 0, log_prob - old[chunk], 0)
            ratio = log_ratio.exp()
            a = adv[chunk]
            pg = -torch.minimum(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a)
            kl_base = NoisePolicy.kl_to_standard_normal(mean, log_std)
            # Normalized by the global number of live decisions per minibatch, so gradients
            # averaged over ranks weigh every decision equally.
            scale = args.minibatches / total * ddp.world
            loss = torch.where(m > 0, pg + args.kl_coef * kl_base, 0).sum() * scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ddp.average_grads(noise_policy)
            grad_norm = torch.nn.utils.clip_grad_norm_(noise_policy.parameters(), args.max_grad_norm)
            # A non-finite loss or gradient on any rank skips the step on all ranks.
            if bool(ddp.sum((~(loss.isfinite() & grad_norm.isfinite())).float().reshape(1)) > 0):
                stats["skipped_steps"] += 1
                optimizer.zero_grad(set_to_none=True)
                stats["updates"] += 1
                continue
            optimizer.step()
            with torch.no_grad():
                epoch_kl += torch.stack([(((ratio - 1) - log_ratio) * m).sum(), m.sum()])
                stats["loss"] += float(loss) / ddp.world
                stats["clip_frac"] += float((((ratio - 1).abs() > args.clip).float() * m).sum())
                stats["kl_base"] += float((kl_base * m).sum())
            stats["updates"] += 1
            # Same value on every rank, so all stop together.
            kl_sum, seen = ddp.sum(epoch_kl.clone())
            stats["approx_kl"] = float(kl_sum / seen.clamp_min(1.0))
            if stats["approx_kl"] > args.target_kl:
                stop = True
                break
        if stop:
            break
    epochs = stats["updates"] / args.minibatches
    for key in ("clip_frac", "kl_base"):
        stats[key] = float(ddp.sum(torch.tensor(stats[key], device=mask.device))) / float(total) / max(epochs, 1e-9)
    stats["epochs"] = epochs
    del stats["updates"]
    return stats


def summarize(batch: dict, ddp: Distributed, prefix: str) -> dict:
    keys = ("returns", "lifted", "entered", "placed", "fallen", "dropped", "diverged", "final_dist", "min_dist")
    values = torch.stack([batch[k].float().sum() for k in keys] + [torch.tensor(float(len(batch["returns"])), device=batch["returns"].device)])
    values = ddp.sum(values)
    count = float(values[-1])
    names = ("return", "lift_rate", "in_box_rate", "success_rate", "fall_rate", "drop_rate", "diverged_rate", "final_dist",
             "min_dist")  # fmt: skip
    return {f"{prefix}/{name}": float(v) / count for name, v in zip(names, values[:-1])}


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    ddp = Distributed()
    torch.manual_seed(args.seed + 1000 * ddp.rank)
    rollouts = Rollouts(args, ddp)
    diffusion = rollouts.diffusion
    noise_policy = NoisePolicy(diffusion.n_tokens, diffusion.n_emb, diffusion.horizon, diffusion.action_dim,
                               hidden=args.hidden, init_log_std=args.init_log_std).to(ddp.device)  # fmt: skip
    optimizer = torch.optim.Adam(noise_policy.parameters(), lr=args.lr)
    start = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=ddp.device, weights_only=False)
        noise_policy.load_state_dict(state["noise_policy"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["iteration"] + 1
    ddp.broadcast(noise_policy)
    log = None
    if ddp.rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "args.json").write_text(json.dumps(asdict(args), default=str, indent=2))
        log = (args.out_dir / "metrics.jsonl").open("a")
        print(f"DSRL-GRPO: {ddp.world} GPU x {args.groups} groups x {args.group_size} envs, "
              f"{len(rollouts.train_rows)} train / {len(rollouts.val_rows)} val grasp starts", flush=True)  # fmt: skip
    rng = torch.Generator().manual_seed(args.seed + 7919 * ddp.rank)
    group = args.group_size
    best = (-1.0, -1.0)

    def evaluate(with_base: bool) -> dict:
        n = args.groups * group
        rows = torch.tensor(rollouts.val_rows).repeat(-(-n // len(rollouts.val_rows)))[:n]
        out = {}
        for mode in ("mean",) + (("base",) if with_base else ()):
            batch = rollouts.run(noise_policy, rows, mode)
            out.update(summarize(batch, ddp, f"eval_{mode}"))
        return out

    for it in range(start, args.iterations):
        tic = time.perf_counter()
        pick = torch.randint(len(rollouts.train_rows), (args.groups,), generator=rng)
        rows = torch.tensor(rollouts.train_rows)[pick].repeat_interleave(group)
        batch = rollouts.run(noise_policy, rows, "sample")
        rollout_s = time.perf_counter() - tic
        adv, informative = group_advantages(batch["returns"], group)
        stats = update(noise_policy, optimizer, batch, adv, args, ddp)
        record = {
            "iteration": it, "rollout_s": rollout_s, "update_s": time.perf_counter() - tic - rollout_s,
            "informative_groups": float(ddp.sum(informative.sum()) / (args.groups * ddp.world)),
            "std": float(noise_policy.log_std.detach().clamp_min(noise_policy.min_log_std).exp().mean()),
            **summarize(batch, ddp, "train"), **stats,
        }  # fmt: skip
        del batch
        # Also at the first iteration of a run, so a resumed run has its own reference point.
        if args.eval_every and (it % args.eval_every == 0 or it == start or it == args.iterations - 1):
            record.update(evaluate(with_base=it == start))
        if ddp.rank == 0:
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            state = {
                "noise_policy": noise_policy.state_dict(), "optimizer": optimizer.state_dict(),
                "iteration": it, "args": asdict(args),
                "diffusion_checkpoint": str(args.diffusion_checkpoint),
                "tracker_checkpoint": str(args.tracker_checkpoint),
            }  # fmt: skip
            # Never overwrite a good checkpoint with a diverged policy: stop instead.
            if not all(bool(v.isfinite().all()) for v in state["noise_policy"].values()):
                raise RuntimeError(f"Non-finite noise-policy parameters at iteration {it}; last.pt kept")
            if it % args.save_every == 0 or it == args.iterations - 1:
                torch.save(state, args.out_dir / "last.pt")
            score = (record.get("eval_mean/success_rate", -1.0), record.get("eval_mean/return", -1.0))
            if "eval_mean/success_rate" in record and score > best:
                best = score
                torch.save(state, args.out_dir / "best.pt")
    if ddp.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
