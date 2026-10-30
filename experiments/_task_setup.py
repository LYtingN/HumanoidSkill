"""Task construction and callbacks used by the experiment entry points."""

from __future__ import annotations

import dataclasses

from can_pnp.config import CanPnpSceneCfg, SceneGeometryCfg
from can_pnp.env_cfg import CanPnpEnvCfgArgs, make_env
from can_pnp.grasp_start import GraspStarts, apply_grasp_start
from can_pnp.layout import layout_term
from can_pnp.task import TaskCfg, TaskTracker


def task_cfg(args) -> TaskCfg:
    return TaskCfg(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TaskCfg) if hasattr(args, f.name)})


def rollout_dependencies(args):
    """Pass the original scene, starts and scoring dependencies to the core."""
    def env_factory(n, seed, device):
        return make_env(
            CanPnpEnvCfgArgs(
                # Episodes are timed here, not by the env: its time-out would reset envs mid-rollout.
                num_envs=n, randomize=False, camera=True, rgb_backend=args.rgb_backend, episode_length_s=1e6,
                seed=seed,
                scene=CanPnpSceneCfg(
                    geometry=SceneGeometryCfg(can_mass=args.can_mass),
                    camera=dataclasses.replace(CanPnpSceneCfg().camera, width=args.render_size[0], height=args.render_size[1]),
                ),
            ),
            device=device,
        )  # fmt: skip

    return dict(env_factory=env_factory, starts_factory=GraspStarts, task_cfg=task_cfg,
                layout_term=layout_term, apply_grasp_start=apply_grasp_start, task_tracker=TaskTracker)
