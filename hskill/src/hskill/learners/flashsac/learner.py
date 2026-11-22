"""Off-policy FlashSAC updates at FM replan boundaries.

Under torch.distributed every rank holds an identical learner replica and its own replay shard.
Gradients and batch-norm statistics are averaged across ranks, so all GPUs collect and learn.
"""

import copy
import math

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .network import FlashSACActor, FlashSACDoubleCritic, FlashSACTemperature, prior_pre_tanh_std
from .optimizer import Network
from .reward_normalization import RewardNormalizer
from .update import update_actor, update_critic, update_temperature


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class AveragedGradients:
    """Optimizer facade that averages gradients over the process group before each step."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        if _distributed():
            # Every replica runs the same code, so the set of parameters with gradients agrees.
            grads = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
            if grads:
                flat = _flatten_dense_tensors(grads)
                dist.all_reduce(flat)
                flat /= dist.get_world_size()
                for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
                    grad.copy_(synced)
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)


def _batch_norms(module):
    return [m for m in module.modules() if hasattr(m, "running_mean")]


@torch.no_grad()
def _average_buffers(module):
    if not _distributed():
        return
    buffers = [b for m in _batch_norms(module) for b in (m.running_mean, m.running_var)]
    flat = _flatten_dense_tensors(buffers)
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    for buffer, synced in zip(buffers, _unflatten_dense_tensors(flat, buffers)):
        buffer.copy_(synced)


@torch.no_grad()
def _broadcast_module(module, src=0):
    if _distributed():
        for tensor in [*module.parameters(), *module.buffers()]:
            dist.broadcast(tensor.data, src)


class Replay:
    """CPU ring storing actual FM decisions, including their pre-reset successors."""

    def __init__(self, capacity, storage_dtypes=None):
        self.capacity = capacity
        self.size = self.cursor = 0
        self.fields = {}
        # Optional per-field storage dtype (e.g. BF16 for values that are BF16 already); samples
        # come back in the dtype the field was added with.
        self.storage_dtypes = dict(storage_dtypes or {})
        self.dtypes = {}

    def add(self, transition):
        count = len(transition["reward"])
        if not 0 < count <= self.capacity:
            raise ValueError("Replay must fit at least one collector batch")
        if not self.fields:
            self.dtypes = {k: v.dtype for k, v in transition.items()}
            self.fields = {k: torch.empty((self.capacity, *v.shape[1:]), dtype=self.storage_dtypes.get(k, v.dtype))
                           for k, v in transition.items()}
        ids = (self.cursor + torch.arange(count)) % self.capacity
        for key, value in transition.items():
            self.fields[key][ids] = value.detach().to(self.fields[key].dtype).cpu()
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size, device):
        ids = torch.randint(self.size, (batch_size,))
        return {k: v[ids].to(device).to(self.dtypes[k]) for k, v in self.fields.items()}

    def state_dict(self):
        # Clone a partial ring so torch.save does not serialize the unused storage.
        fields = {k: v[:self.size].clone() if self.size < self.capacity else v for k, v in self.fields.items()}
        return dict(capacity=self.capacity, size=self.size, cursor=self.cursor, fields=fields,
                    dtypes={k: str(v).removeprefix("torch.") for k, v in self.dtypes.items()})

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError(f"Replay capacity {state['capacity']} differs from {self.capacity}")
        self.size, self.cursor = state["size"], state["cursor"]
        self.fields = {}
        self.dtypes = {k: getattr(torch, v) for k, v in state.get("dtypes", {}).items()}
        for key, value in state["fields"].items():
            self.fields[key] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            self.fields[key][:len(value)] = value
            self.dtypes.setdefault(key, value.dtype)


class FlashSAC:
    def __init__(self, obs_dim, state_dim, action_dim, device, gamma=0.99,
                 actor_width=128, critic_width=256, blocks=2, noise_scale=None, target_std=.15):
        self.device = torch.device(device)
        self.gamma = gamma
        # With noise_scale, the actor's tanh action times noise_scale is the FM latent; the actor
        # starts at the observation-independent law with the FM prior's unit second moment.
        prior_std = None if noise_scale is None else prior_pre_tanh_std(noise_scale)
        self.config = dict(obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
                           actor_width=actor_width, critic_width=critic_width, blocks=blocks, gamma=gamma,
                           noise_scale=noise_scale, prior_std=prior_std, target_std=target_std)
        self.action_dim = action_dim
        self.bins = 101

        def optimized(module, normalize):
            module.to(self.device)
            _broadcast_module(module)  # identical replicas even if ranks were seeded differently
            optimizer = AveragedGradients(torch.optim.AdamW(module.parameters(), lr=3e-4, weight_decay=0.0))
            net = Network(module, optimizer, use_weight_normalization=normalize)
            if normalize:
                net.normalize_parameters()
            return net

        self.actor = optimized(FlashSACActor(blocks, obs_dim, actor_width, action_dim, prior_std), True)
        self.critic = optimized(FlashSACDoubleCritic(blocks, state_dim + action_dim, critic_width,
                                                     self.bins, -5., 5.), True)
        self.target = Network(copy.deepcopy(self.critic.network).requires_grad_(False),
                              ema_source=self.critic, ema_tau=0.01)
        # The actor log density is summed over a full 32 x 47 noise trajectory.
        self.temperature = optimized(FlashSACTemperature(0.01 / action_dim), False)
        self.target_entropy = 0.5 * action_dim * math.log(2 * math.pi * math.e * target_std**2)
        self.normalizer = RewardNormalizer(gamma, 5., False, self.device)
        self.updates = 0
        self.calibrated = False
        self.noise = self.remaining = None
        self.repeat_weights = torch.arange(1, 17, device=device, dtype=torch.float).pow(-2)

    @torch.no_grad()
    def act(self, observation, reset):
        mean, std = self.actor.network.get_mean_and_std(observation, training=False)
        if self.remaining is None or len(self.remaining) != len(mean):
            self.remaining = torch.zeros_like(reset, dtype=torch.long)
            self.noise = torch.zeros_like(mean)
        refresh = reset.bool() | (self.remaining == 0)
        repeats = torch.multinomial(self.repeat_weights, len(mean), replacement=True) + 1
        self.remaining = torch.where(refresh, repeats, self.remaining) - 1
        self.noise = torch.where(refresh[:, None], torch.randn_like(mean), self.noise)
        return torch.tanh(mean + std * self.noise)

    @torch.no_grad()
    def act_deterministic(self, observation):
        """The actor's mean action (as at deployment), without exploration noise."""
        mean, _ = self.actor.network.get_mean_and_std(observation, training=False)
        return torch.tanh(mean)

    @torch.no_grad()
    def calibrate(self, batch):
        """Set batch-norm running statistics from replay before the first update.

        ``act`` and the actor loss's critic use running statistics. Without this they would start
        from mean 0 / variance 1 and approach the data at momentum 0.01 per update.
        """
        actor_obs = torch.cat([batch["actor_observation"], batch["actor_next_observation"]])
        for network, kwargs in ((self.actor, dict(observations=actor_obs)),
                                (self.critic, dict(observations=batch["observation"], actions=batch["action"]))):
            norms = _batch_norms(network.network)
            momenta = [m.momentum for m in norms]
            for m in norms:
                m.momentum = 1.0
            network.network(**kwargs, training=True)
            for m, momentum in zip(norms, momenta):
                m.momentum = momentum
            _average_buffers(network.network)
        self.target.network.load_state_dict(self.critic.network.state_dict())
        self.calibrated = True

    def update(self, batch):
        batch["reward"] = self.normalizer.normalize_rewards(batch["reward"])
        kwargs = dict(device=self.device, use_amp=False, grad_scaler=None)
        info = {}
        if self.updates % 2 == 0:
            info = update_actor(self.actor, self.critic, self.temperature, batch, 0., **kwargs)
            info.update(update_temperature(self.temperature, info["actor/entropy"], self.target_entropy))
            _average_buffers(self.actor.network)
        info.update(update_critic(self.actor, self.critic, self.target, self.temperature, batch,
                                  -5., 5., self.bins, self.gamma, 1, **kwargs))
        _average_buffers(self.critic.network)
        # The target runs in training mode; its running statistics are unused but kept equal.
        _average_buffers(self.target.network)
        self.target.ema_update_parameters()
        self.updates += 1
        return {k: float(v.detach()) for k, v in info.items()}

    def checkpoint(self):
        rms = self.normalizer.G_rms
        return dict(algorithm="flashsac", config=self.config, updates=self.updates, calibrated=self.calibrated,
                    actor=self.actor.network.state_dict(), critic=self.critic.network.state_dict(),
                    target=self.target.network.state_dict(), temperature=self.temperature.network.state_dict(),
                    optimizers={k: getattr(self, k).optimizer.state_dict()
                                for k in ("actor", "critic", "temperature")},
                    normalizer=dict(G_r_max=self.normalizer.G_r_max, mean=rms.mean, var=rms.var, count=rms.count))

    def load_checkpoint(self, ckpt):
        if ckpt.get("algorithm") != "flashsac" or "optimizers" not in ckpt:
            raise ValueError("Resuming needs a full FlashSAC training checkpoint")
        if ckpt["config"] != self.config:
            raise ValueError(f"FlashSAC configuration differs: {ckpt['config']} vs {self.config}")
        for key in ("actor", "critic", "target", "temperature"):
            getattr(self, key).network.load_state_dict(ckpt[key])
        for key, state in ckpt["optimizers"].items():
            getattr(self, key).optimizer.load_state_dict(state)
        norm, rms = ckpt["normalizer"], self.normalizer.G_rms
        self.normalizer.G_r_max = norm["G_r_max"].to(self.device)
        rms.mean, rms.var, rms.count = (norm[k].to(self.device) for k in ("mean", "var", "count"))
        # Episodes restart on resume, so their running discounted returns restart too.
        self.normalizer.G_r = torch.zeros(1, device=self.device)
        self.updates, self.calibrated = ckpt["updates"], ckpt["calibrated"]
        self.noise = self.remaining = None
