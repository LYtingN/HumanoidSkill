# nyx 数据与权重目录

来源是本机 `Downloads/residual_rl.zip` 和 `Downloads/downloads.zip`。包内目录结构保留，实际数据放在 CFS，开发机代码目录通过软链接访问。

| 压缩包 | CFS 解压位置 | 开发机代码中的入口 |
|---|---|---|
| `residual_rl.zip` | `/mnt/cfs/usr/nyx/data/residual_rl` | `/root/workspace/HumanoidSkill/data/residual_rl` |
| `downloads.zip` | `/mnt/cfs/usr/nyx/checkpoints/downloads` | `/root/workspace/HumanoidSkill/downloads` |

`/root/workspace/HumanoidSkill/checkpoints` 也链接到 CFS 的 `checkpoints`。训练任务使用下面的 CFS 绝对路径，不依赖开发机软链接或重新制作镜像。

## 数据集

`residual_rl` 下的 `can_pnp`、`carry_plate`、`push_whiteboard` 各有 250 个 episode，包含 parquet、视频及 meta 信息。`can_pnp` 还包含 `derived/keypoints_v1` 的 250 份关键点标签。

从开发机的仓库根目录使用以下相对路径：

```text
data/residual_rl/can_pnp/data/chunk-000/episode_000000.parquet
```

对应的任务容器路径：

```text
/mnt/cfs/usr/nyx/data/residual_rl/can_pnp/data/chunk-000/episode_000000.parquet
```

这些关键点 NPZ 包含 `pos_w`、`quat_w` 等字段，不是 SuperMimic MotionLoader 所需的完整运动 NPZ，也不是 compact motion library。因此不能把 `derived/keypoints_v1` 填入 SuperMimic 的 `npz_dir`；从头训练 SuperMimic 仍需另备其运动数据。

## 当前 can-pnp 配置使用的资产

| 用途 | `/mnt/cfs/usr/nyx/checkpoints/downloads/` 下的相对路径 |
|---|---|
| SuperMimic tracker | `supermimic/supermimic_0100000.pt` |
| 默认 SuperMimic flow matching policy | `supermimic-fm/can_pnp_supermimic_flow_matching_best_ema.pt` |
| 推理源码，包含当前 loader 所需模块 | `ODE/inference_source` |
| 另存的 SONIC DDIM 权重，当前 can-pnp 不兼容 | `supermimic-dp/diffusion_policy.pt` |

`ODE/inference_source` 中的 `load_policy` 会根据 checkpoint 的 `policy_type` 选择模型，也包含旧 DDIM 与 flow matching 的加载实现。这里复用的是归档里的推理源码快照；不把它当作完整的外部训练仓库。

本机 `profiles.local.json` 中的 can-pnp DSRL / v3 入口使用上述 37 维 flow matching 权重和 tracker。实际检查发现，`supermimic-dp/diffusion_policy.pt` 的目标是 `action.wbc`，只有 31 维且没有根部旋转；当前 can-pnp loader 会明确拒绝它，因此不将它作为默认输入。其他实验文件保持原样：

- `morphmimic/`：两个 MorphMimic tracker、源码及验证记录。
- `5points_FM/`：五点 FM 权重及源码。
- `ODE/`：ODE 权重及其带 carry 的推理实现。
- `DSRL_finetune_20261116/`：已有实验的 best / last 检查点和训练记录。

这些模型有不同接口；没有把 MorphMimic、五点 FM、ODE 或旧 v4 微调检查点自动设为当前 can-pnp 的恢复训练输入。归档内记录的旧服务器绝对路径用于保留来源，实际运行参数应使用本机配置中的新 CFS 路径。

为兼容现有 README，CFS `checkpoints` 下还提供 `supermimic_0100000.pt`、`diffusion_policy.pt` 和 `can_pnp_supermimic_flow_matching_best_ema.pt` 的软链接。项目根目录另有 `supermimic_0100000.pt` 入口。

## 导入记录

归档 SHA256：

```text
03e2be6ebe87daecd2cd7149ee9c76050f4944b5cac8ddc682dea4e698fb4304  residual_rl.zip
4034d0f84d94f0d4f225ad0140cd5f5325617caac3968e2b0d8ca8060fcc5465  downloads.zip
```

导入清单和 CPU 加载检查结果放在 `/mnt/cfs/usr/nyx/imports/20261119-datasets/`。导入会核对远端归档 SHA256、逐文件 ZIP CRC 和解压后的文件数量；验证不启动训练、不申请 GPU。
