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

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import tyro

from can_pnp.dsrl import NoisePolicy
from can_pnp.rl.grpo import Distributed, group_advantages, summarize, update
from can_pnp.sim.rollouts import Rollouts


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


def main(args: Args | None = None) -> None:
    args = args or tyro.cli(Args)
    ddp = Distributed()
    torch.manual_seed(args.seed + 1000 * ddp.rank)
    rollouts = Rollouts(args, ddp.device, ddp.rank)
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
