# Dynamic Query + 共享 KV Cache：S1/S0 双向通道与交替联合训练

> 状态：idea 记录（2026-11-20），**可行性未验证**，已经讨论过一轮并做了修订。
> 与 [IDEA_WM_IDM_BRIDGE.md](IDEA_WM_IDM_BRIDGE.md) 的关系：给"S1/S0 级联联合优化"提供一种具体的接口和训练机制。

---

## 1. 原始想法

1. **Pretrain 阶段**：在 S1（FM / DiT）里加入一组 **dynamic query** token，但在 pretrain 时把它们 **mask 掉**，不影响原有的 flow matching 训练。
2. **Post-train 阶段**：**打开** dynamic query，用来学 action 的结构，以及当前架构内部的结构。
3. **S1 的输出改成 5 个 keypoint**，交给 tracker（S0）。
4. **S0 加一个可训练 head**，这个 head **读取 DiT 的 KV cache**。
5. **维护一个 KV cache buffer**，用它**交替更新 S1 和 S0**。

## 2. 修订后的理解：两个方向的通道

| 通道 | 方向 | 作用 |
|---|---|---|
| **5 keypoints** | S1 → S0（显式） | 稀疏、任务关键的硬目标，例如手的位置 |
| **KV cache** | S1 → S0（隐式） | 让 S0 理解 **S1 的全身 motion 意图和任务**。5 个点不足以表达全身意图，KV 补上缺失的部分 |
| **dynamic query** | S0 → S1（向上） | 让 DiT 理解**下层的动力学**，也就是 S0 实际能执行成什么样 |

```text
            ┌──────────── dynamic queries（理解 S0 动力学，向上）◄─── S0 relabel：Φ(r)
            ▼
o ──► S1 DiT (FM) ──► 5 keypoints（硬目标）──────────────┐
        │                                                ▼
        └── KV cache（全身意图 + 任务，向下）──► S0 head ──► SuperMimic ──► PD
```

## 3. Problem

- **显式接口无法同时做到"稀疏"和"完整"**：
  - 32 帧全身 reference 是过约束，要求 S0 跟住所有关节；
  - 5 个 keypoint 又不足以表达全身意图。
- **S1 不了解 S0 的动力学**：它以为参考会被理想执行，即 r ≈ Φ(r)。
- **S0 不了解 S1 的意图和任务**：它对所有部位、所有时刻的跟踪一视同仁。

## 4. Novelty（可能的点）

- **双向通道**：
  - 向下，硬 keypoint 加软 KV latent，S0 负责在物理可行的前提下补全全身动作；
  - 向上，dynamic query 让 S1 感知 S0 的动力学。
- **随 S0 共同演化的动力学感知**：dynamic query 的监督信号来自**当前 S0** 的执行结果 Φ(r)，每轮交替都重新标注。
  - 与 HWAM 的区别：HWAM 的 state 标签来自固定的演示数据，对应的是数据采集时的控制器。
- **pretrain 时 mask 有明确理由**：pretrain 的演示数据记录的是 SONIC/Isaac 的动力学，不是 S0（SuperMimic/MuJoCo）的，所以 pretrain 时不应该学这部分；post-train 时用 S0 回放标注的数据把它打开。
- **级联梯度**：S0 的 PPO 损失对其输入的 KV 是可微的，梯度可以直接回传到 S1 的可训练参数，**不需要对物理仿真求导**。

## 5. Method（初步设想）

**S1**：骨干冻结，加 K 个零初始化的 dynamic query。query 的训练目标是：在给定 keypoint 计划的条件下，预测 S0 实际执行的结果 Φ（手部 / 根部 / 关键 keypoint 的轨迹）。动作头可以把 query 当作条件。

**S0**：SuperMimic 冻结，加一个零初始化的残差 head，这个 head 读取 S1 的 KV。

训练分阶段：

- **阶段 A（KV → 全身 motion）**：在现有的跟踪环境里训练。KV 由 S1 在演示观测上离线预先算好。输入是 5 个 keypoint 加 KV，**奖励是全身跟踪**。由于 keypoint 本身不足以完成全身跟踪，S0 head 被迫从 KV 里读取全身意图。这个阶段不需要渲染相机。
- **阶段 B（交替联合优化，在 can-pnp 环境里闭环）**：
  1. **S0 步**：固定 S1。S1 每 8 步在闭环里推理一次，KV 写入 buffer。S0 head 用"keypoint 跟踪 + 任务奖励"做 RL，全身部分只作为软正则。可以选择把梯度经过 KV 回传到 S1 的可训练参数。
  2. **S1 步**：固定 S0。用当前 S0 回放并重新标注 Φ，然后训练 dynamic query（预测 Φ）和动作头（做任务对齐）。
  3. 刷新 buffer，重复。

## 6. 风险与待确认

- **训练信号冲突**：如果 query 的 KV 也被 S0 读到，那么 query 同时受到两种梯度：向上的"预测 Φ"，和向下的"S0 回传的梯度"。可能需要 S0 只读骨干的 KV，或者对回传的梯度做截断或加权。
- **buffer 与场景脱节**：KV 编码的是旧观测下的意图，所以阶段 B 中 S1 必须在闭环里，buffer 只用于小范围的 off-policy 复用。
- **捷径**：S0 可能绕过 keypoint、完全依赖 KV。keypoint 的跟踪奖励权重必须足以保证它是硬约束。
- **5 个 keypoint 具体是哪 5 个**（例如 root/head + 双手 + 双脚？）。夹爪开合放在哪里？SuperMimic 需要重训，以支持"稀疏 keypoint 加 latent"作为输入。
- **KV 取自哪一步**：FM 有多个去噪步，KV cache 取自哪个 τ？
- **耦合**：S0 绑定在这个 S1 上，丧失了可替换性。论文里要正面说明。
- **控制变量**：输出改成 keypoint、加 KV 通道、加 query 是三个独立变量，实验需要分开消融。

## 7. 先做的廉价验证

- **oracle 意图上界**：给 S0 残差 head 喂 oracle 意图（阶段标签、罐子和盒子的位置），观察 can-pnp 的放置成功率（目前 CLOSE 起点 0/8）能否提升。如果没有提升，说明瓶颈不在"S0 不知道意图"。
- **KV 中的全身信息**：从冻结的 S1 KV 线性 probe 全身 32 帧 reference，看误差是多少。这能验证"KV 包含足够的全身 motion 信息"这一前提。
- **query 的价值**：比较 query 预测的 Φ 和实际执行的 Φ 的误差，以及以 query 为条件后，S1 规划的可执行性（手部误差对比罐子半径 2.75 cm）是否提升。
