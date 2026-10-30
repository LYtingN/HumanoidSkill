"""GRPO on the DDIM noise of the frozen diffusion policy (shared by sim and imagined rollouts).

Advantages are returns normalized within each group of consecutive samples (zero for constant
groups). The update is PPO-clip on the noise log-likelihood plus ``kl_coef`` x KL to N(0, I),
with an early stop once an epoch's approx KL passes ``target_kl``. ``cfg`` is any object with
``epochs``, ``minibatches``, ``clip``, ``kl_coef``, ``target_kl`` and ``max_grad_norm``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from can_pnp.dsrl import NoisePolicy


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



def group_advantages(returns: torch.Tensor, group_size: int,
                     min_std: float | torch.Tensor = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """Returns normalized within each group of consecutive envs; zero for groups whose spread is
    at most ``min_std`` (a float, or (groups, 1) per group)."""
    r = returns.view(-1, group_size)
    std = r.std(dim=1, keepdim=True)
    informative = std > min_std
    adv = torch.where(informative, (r - r.mean(dim=1, keepdim=True)) / (std + 1e-6), torch.zeros_like(r))
    return adv.flatten(), informative.flatten().float()  # (envs,), (groups,)


def update(noise_policy: NoisePolicy, optimizer: torch.optim.Optimizer, batch: dict, adv: torch.Tensor,
           cfg, ddp: Distributed) -> dict:  # fmt: skip
    d, n = batch["mask"].shape
    cond = batch["cond"].flatten(0, 1)
    noise, old = batch["noise"].flatten(0, 1), batch["log_prob"].flatten(0, 1)
    mask = batch["mask"].flatten()
    adv = adv[None].expand(d, n).flatten()
    total = ddp.sum(mask.sum().clone()).clamp_min(1.0)
    stats = {"loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "kl_base": 0.0, "updates": 0, "skipped_steps": 0}
    stop = False
    for _ in range(cfg.epochs):
        order = torch.randperm(len(mask), device=mask.device)
        epoch_kl = torch.zeros(2, device=mask.device)  # summed KL, live decisions seen
        for chunk in order.chunk(cfg.minibatches):
            m = mask[chunk]
            mean, log_std = noise_policy(cond[chunk])
            log_prob = NoisePolicy.log_prob(noise[chunk], mean, log_std)
            # Masked decisions get ratio 1, so no exp of their (meaningless) log-ratio enters backward.
            log_ratio = torch.where(m > 0, log_prob - old[chunk], 0)
            ratio = log_ratio.exp()
            a = adv[chunk]
            pg = -torch.minimum(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)
            kl_base = NoisePolicy.kl_to_standard_normal(mean, log_std)
            # Normalized by the global number of live decisions per minibatch, so gradients
            # averaged over ranks weigh every decision equally.
            scale = cfg.minibatches / total * ddp.world
            loss = torch.where(m > 0, pg + cfg.kl_coef * kl_base, 0).sum() * scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            ddp.average_grads(noise_policy)
            grad_norm = torch.nn.utils.clip_grad_norm_(noise_policy.parameters(), cfg.max_grad_norm)
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
                stats["clip_frac"] += float((((ratio - 1).abs() > cfg.clip).float() * m).sum())
                stats["kl_base"] += float((kl_base * m).sum())
            stats["updates"] += 1
            # Same value on every rank, so all stop together.
            kl_sum, seen = ddp.sum(epoch_kl.clone())
            stats["approx_kl"] = float(kl_sum / seen.clamp_min(1.0))
            if stats["approx_kl"] > cfg.target_kl:
                stop = True
                break
        if stop:
            break
    epochs = stats["updates"] / cfg.minibatches
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
