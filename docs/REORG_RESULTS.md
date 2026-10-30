# 代码布局迁移验收

迁移分支：`refactor/layout`。原始提交：`dcef958bb45dd4644ba16509b905909124f1b7d8`。
按任务要求分六次提交，未 push，未操作远端服务器。

## 实际迁移

文件按 `CODEX_REORG_TASK.md` 的对照表迁移，以下是具体选择和补充：

- 机器人常量放在 `hskill/embodiment/g1_aloha.py`，构建函数放在同目录 `robot.py`。
  机器人资产随 embodiment 移动；共享的 `srgb` 函数放在 `hskill/utils/colors.py`。
- `pipeline.py` 拆为 `hskill/s0/controller.py` 和 `hskill/s1/sources.py`；
  `body/imagine.py` 放在 `hskill/executors/imagine.py`，`rl/imagine.py` 改为 `executors/imaginer.py`。
- `Episodes` 数据类单独放在 `hskill/data/episodes.py`，buffer 直接引用它。
  数据类字段和默认值不变。
- `experiments/_task_setup.py` 保留原有的环境和任务配置构造，向核心传入环境工厂、
  grasp-start 工厂及应用函数、layout 查询、任务配置及 tracker、验证集 episode 编号。
  验证集编号随任务放在 `can_pnp/starts.py`，数值不变。没有新增抽象基类或协议。
- 原 `test_v3.py` 的任务测试和 `test_body_wm.py` 的场景集成测试放入任务 tests；
  其余核心测试移入 hskill。测试名字和总数仍为原有 21 项。
- `wheels/`、`.python-version` 和锁文件移至 workspace 根，新 `.venv` 由
  `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple uv sync --all-extras --offline`
  创建。锁文件中 164 个第三方包的版本与原锁文件完全一致。
- 原 `can-pnp/` 的受版本管理代码全部迁走。由于要求不能移动或删除已有日志，
  本地保留 `can-pnp/logs/`；噪声策略仍使用原路径。旧环境在验收后清理，活动环境在根目录。

`supermimic-reproduced/`、Humanoid-Diffusion、已有 data/logs/checkpoint 均保持原位和内容。
body world model 的新文件相对 workspace 的深度相同，因此 `DEFAULT_REPO` 的
`parents[4]` 表达式仍正确；已验证它解析到本仓库的 `supermimic-reproduced/`。
四个实验入口的 CLI 字段、类型和默认值逐项比对，只有允许调整的 `diffusion_repo` 默认路径变化。

## TODO(decouple)

无。任务依赖均通过参数和回调传入，没有因解耦失败而留在任务包中的算法模块。
核心仍保留原有的 env 字段访问和数值逻辑，未设计新的环境或任务接口。

## 验收结果

初次在沙箱内执行时 CUDA 不可访问，测试为 20 passed、1 skipped，三个 GPU 命令在初始化失败。
之后按原 README 的 NVIDIA 570 库设置，在可访问 GPU 的沙箱外补验。
迁移前代码从原始提交提取至 `/tmp/hskill-reorg-original/`，使用旧环境；
迁移后使用根目录新环境。所有新验收输出写入 `/tmp/`，没有写入既有训练日志。

| 项目 | 迁移前 | 迁移后 |
|---|---|---|
| 1. 全部测试 | GPU：21 passed | GPU：21 passed |
| 2. 依赖方向 | 原结构任务与算法混合 | `grep -rn "can_pnp" hskill/` 输出为空 |
| 3. DSRL 恢复冒烟 | 从 iteration 600 恢复，601 完成 | 同样完成 iteration 601；指标同量级，详见下表 |
| 4. v3 冒烟 | 指定参数，cycle 0–2 完整完成 | 同样完成 cycle 0–2，无异常退出 |
| 5. Φ 精度 | 1 episode × 16 env，完成 | 同样完成；Φ 七类误差全预测步最大相对变化 9.33%，同量级；均无跌倒 |
| 6. 可视化 | 原入口 `--help` 成功 | `experiments.view_policy --help` 和 `can-pnp-run --help` 成功；未启动交互窗口 |
| 7. 启动脚本 | 原端口 29531 / 29541 | `bash -n` 通过；静态 rendezvous、端口及入口路径检查通过 |

DSRL 两次单 GPU 运行的 `train/*` 指标如下。同量级，但没有逐位一致；随机种子、
物理参数、模型、优化器与训练逻辑均未调整。

| 指标 | 迁移前 | 迁移后 |
|---|---:|---:|
| return | 7.261702 | 7.849232 |
| lift_rate | 1.00000 | 1.00000 |
| in_box_rate | 0.78125 | 0.81250 |
| success_rate | 0.71875 | 0.78125 |
| fall_rate | 0 | 0 |
| drop_rate | 0.03125 | 0.06250 |
| diverged_rate | 0 | 0 |
| final_dist (m) | 0.108877 | 0.140830 |
| min_dist (m) | 0.082336 | 0.083084 |

Φ 第 32 步的误差如下；第 1–32 步的七项 Φ 误差最大相对变化为 9.33%。

| Φ 指标 | 迁移前 | 迁移后 |
|---|---:|---:|
| root_pos_cm | 0.761674 | 0.772923 |
| joint_rms_rad | 0.013242 | 0.013408 |
| gripper_pos_cm | 1.711967 | 1.790512 |
| gripper_pos_torso_cm | 1.467039 | 1.545099 |
| gripper_rot_deg | 2.985537 | 3.083443 |
| camera_pos_cm | 1.036694 | 1.069709 |
| camera_rot_deg | 1.121023 | 1.132893 |

完整误差表和 reference 对照保留在两次运行的 `body_wm/base_base.json`。

原有 91 个资产文件逐个校验，内容完全一致。对原 205 个函数和方法做 AST 比对：
除 `Rollouts.__init__` 和 `Imaginer.__init__` 的依赖注入构造调整外，其余 203 个在
统一 import 路径、回调引用和允许的类型标注调整后完全一致。
`ruff check --select F821,F822,F823`、Python 语法解析和 `git diff --check` 通过。

本地详细输出：`/tmp/hskill-reorg-baseline-gpu/`、`/tmp/hskill-reorg-post-gpu/`；
各自 `summary.json` 记录完整命令、退出码和耗时。缓存设置为
`TMPDIR=$HOME/.cache/hskill-reorg/tmp`、
`TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/hskill-reorg/torchinductor`。
