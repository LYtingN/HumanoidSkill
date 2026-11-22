# HumanoidSkill：通过 Codex 提交百舸训练

开发机保持无 GPU，用来修改代码、准备环境和数据；训练时创建 AIHC 任务申请 GPU。默认配置是 **1 个 Worker × 8 张 GPU**。任务结束后 GPU 由平台释放，检查点和日志保存在 CFS。

## 文件位置

| 内容 | 位置 |
|---|---|
| 本机代码 | `/home/seer-26101301/Documents/HumanoidSkill` |
| 开发机代码 | `/root/workspace/HumanoidSkill` |
| 开发机 Python | `/root/workspace/HumanoidSkill/deploy/aihc/environment/.venv/bin/python` |
| 数据 / 预训练权重 | `/mnt/cfs/usr/nyx/data`、`/mnt/cfs/usr/nyx/checkpoints` |
| 下载和编译缓存 | `/mnt/cfs/usr/nyx/system_config` |
| 每次训练的代码、参数、输出 | `/mnt/cfs/usr/nyx/work/HumanoidSkill/runs/<任务名>` |
| 本机 AIHC 凭证 | `~/.aihc/nyx.yaml`，权限 `600`，不进入仓库 |

40 GB 是开发机系统盘容量，不是 GPU 显存或内存。代码和环境放系统盘；大数据、模型权重、运行输出放 CFS。用户数据包的目录映射及权重用途见 [DATA.md](DATA.md)。

## 首次配置

本次机器上的 `site.local.json` 已填入 SSH、资源池、队列、CFS 和 GPU 资源信息。其他机器使用时，先复制两个 `*.example.json` 为对应 `*.local.json`。本地配置和生成的任务文件已加入 `.gitignore`。

`site.local.json` 的 `image` 必须指向已完成环境安装的训练镜像，使用固定版本 tag 或 digest。`gpu_resource_name` 使用平台资源键，不能填写显卡显示名。物理资源池与任务 API 路由分别保存在 `pool` 和 `job_api_pool`。

在 `profiles.local.json` 中填写要运行的任务：

- `supermimic`：`library` 为 compact library `.pt`，或填写含运动 `.npz` 的 `npz_dir`。两者均有值时使用 `library`。
- `can-pnp-dsrl` / `can-pnp-v3`：填写 `tracker`、`diffusion` 两个检查点，以及外部 `Humanoid-Diffusion` 源码的 `diffusion_repo`。这些都应是任务能访问的 `/mnt/cfs/...` 路径。外部代码及检查点内部引用的资产也必须备齐；建议固定外部 Git commit。
- `fivepoint-flashsac`：`fm` 为五点 FM 权重（同目录须有 `inference_source/`），`tracker_release` 为 MorphMimic v0.1 发布目录（含 `source/` 和 `morphmimic_policy/`）。工具固定传入 `--auto-resume`；8 个进程各自采样、各存一份 replay，并同步平均梯度，`--batch-size` 是每卡批量。该入口需要 AIHC 环境中的 `onnxruntime` 和 `pyzmq`（v0.1 的 `policy_node` 在导入时使用）。
- 可选 `resources`（`gpu_count`、`cpu`、`memory_gib`、`shared_memory_gib`）和 `priority`（`high` / `normal` / `low`）只覆盖该 profile 的任务资源和优先级，其余沿用 `site.local.json`。
- `args` 是额外训练参数数组，例如 `["--iterations", "100"]`。输出路径和上述数据路径由工具统一传入，避免重复覆盖。

默认申请 48 CPU、500 GiB 内存、250 GiB 共享内存，来自团队八卡任务示例，可以按训练规模修改。它们不是性能保证；队列总空闲卡数也不保证某台机器能提供连续的 8 卡及足够 CPU/内存。

## 日常提交

在本机仓库目录执行：

```bash
python3 deploy/aihc/task.py doctor supermimic
python3 deploy/aihc/task.py prepare supermimic
# prepare 输出本机 runs/<任务名> 路径；检查其中的 manifest.json 和 job.json。
python3 deploy/aihc/task.py submit deploy/aihc/runs/<任务名>
python3 deploy/aihc/task.py status <job-id>
python3 deploy/aihc/task.py list
```

`prepare` 只在本机生成计划，即使资料未齐也会列出缺项；它不会上传或申请 GPU。`submit` 会阻止不完整配置，检查远端 CFS 输入路径，上传代码快照，然后调用一次 `aihc job create`。改过配置后应重新 `prepare`。

可以直接告诉 Codex：

> 用 AIHC 提交 supermimic 八卡训练，使用配置中的数据，运行名和参数按本次要求设置。

或：

> 提交 can-pnp-dsrl 八卡任务，tracker 用 /mnt/cfs/.../tracker.pt，diffusion 用 /mnt/cfs/.../best.pt，外部代码在 /mnt/cfs/.../Humanoid-Diffusion，训练 100 次迭代。

Codex 会读取根目录 `AGENTS.md`，核对输入、生成任务、提交并返回任务 ID。仅说“配置一下”或“查看状态”不会启动训练。

如果 create 超时，服务端仍可能已创建成功。工具会保存提交记录并阻止同一计划重复提交；先按完整任务名查询，不能直接重跑或新建同参数任务。

## 环境与镜像

AIHC 环境独立锁定 PyTorch 2.14.1 / torchvision 0.29.1 的 CUDA 13.0 构建、Warp 1.18.0 和仓库固定的 mjlab commit。代码通过 `PYTHONPATH` 加载；原项目 CUDA 12 环境不受影响。以后项目新增依赖时也要更新 AIHC 环境依赖。

`mjviser` 固定为原仓库使用的 0.0.14，避免新版引入 MuJoCo 4 的二进制包并覆盖 MuJoCo 3.16 文件。timm / diffusers 也与原 can-pnp 锁文件保持一致。已校验 SHA256 的 Linux Python 3.12 PyTorch wheel 另存于 CFS 的 `system_config/wheels/`，供网络异常时复用。

开发机基础镜像为 `ccr-registry.baidubce.com/aihc/aibox-cuda:13.3.1-cudnn-devel-ubuntu24.04`。需要 Python 3.12、uv，以及 `git rsync libegl1 libgl1 libglib2.0-0`。在开发机运行：

```bash
bash /root/workspace/HumanoidSkill/deploy/aihc/setup_environment.sh
```

安装成功后生成 `environment/.ready`，记录已验证环境的锁文件哈希。保存开发机为团队 CCR 镜像，并等镜像制作成功再把完整地址写入 `site.local.json`：

```bash
~/.local/bin/aihc dev commit <dev-id> -C ~/.aihc/nyx.yaml \
  --registry <CCR地址> --namespace seer-eic \
  --image-name nyx-humanoidskill --image-tag <固定版本>
~/.local/bin/aihc dev img-list <dev-id> --image-pack-job-id <返回的制作任务ID> \
  -C ~/.aihc/nyx.yaml
```

注意：CLI 1.6.5 的 `dev img-list` 会把接口状态 `3` 错误显示成 `failed`。应使用本工具读取原始状态：

```bash
python3 deploy/aihc/task.py image-status <镜像制作任务ID>
```

官方状态为：1 初始化、2 运行中、3 制作中、4 推送中、5 完成、30 失败。制作期间开发机可能暂停响应 SSH，不能据此重启或重复创建镜像任务。完成且本机 `image.local.json` 记录匹配时，加 `--activate` 可将镜像地址写入本地站点配置。

镜像包含 SSD 上的代码和环境，CFS 数据不打包入镜像。训练任务把本次代码快照解压到临时工作目录，复用镜像里的 Python 环境，校验环境哈希，再用 torchrun 启动 8 个进程。输出写入对应 CFS run 的 `outputs/`，日志同时保留在 `train.log` 和平台日志中。

环境安装在无 GPU 开发机上可以检查导入和 CLI 参数；真实 CUDA、NCCL、仿真器及八卡性能仍需首次训练任务验证。

## 检查与排障

```bash
python3 -m unittest discover -s deploy/aihc -p 'test_*.py'
```

使用 `status` 获取任务详情和 Pod 名；平台日志命令为 `aihc job logs --podname <Pod名> -p <job_api_pool> -q <queue> -C ~/.aihc/nyx.yaml`。停止已确认的本人任务使用 `aihc job stop <job-id>` 并携带相同 pool、queue、config 参数。

入口不安装驱动、不在 GPU 任务里在线安装大包、不使用保活 sleep。预训练模型可能还有额外下载需求，应先在开发机缓存完整，尤其是外部 diffusion 模型使用的视觉编码器。

接口格式参考：[百度官方 CLI 文档](https://cloud.baidu.com/doc/AIHC/s/Tm7x702fo)。
