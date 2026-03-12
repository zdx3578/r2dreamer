# ARC3 / Atari Structured Dreamer Project Document

## 1. 项目定位

本项目的目标不是继续堆叠一个普通 Dreamer，而是在保留 Dreamer 主干稳定性的前提下，逐步推进为一个**结构化世界模型系统**。该系统面向两类前端输入：

- **Atari 路线**：像素输入，经视觉前端编码后进入统一后端；
- **ARC3 路线**：grid / state 输入，经结构前端编码后进入统一后端。

当前主线优先在 **Atari** 上验证训练稳定性、结构可读性、规则接口的可执行性与多步鲁棒性；ARC3 路线暂时冻结，等待 Atari 主线形成更稳的闭环后再回灌。

---

## 2. 当前总体判断

当前仓库已经不是“结构化 Dreamer + 对象化 + 规则原型监督头”这么简单，而是已经推进为：

**Dreamer 主干 -> 结构读出 -> 对象化过程 -> 规则原型监督层 -> RuleMemory / RuleApply 可执行规则接口 -> 2/4/7-step `rho` shadow rollout -> Atari closed-loop sign-off**

这里必须明确一个当前边界：

> 当前 rollout 仍然是 **`rho-only shadow rollout`**，
> 不是 **full structured rollout**，也不是 **full latent rollout**。

因此，当前系统已经具备 **executable rule path**，但还没有进入：

- full structured rollout
- full latent rollout
- planner 主路径
- actor / value 直接消费规则通道
- 用规则路径替代主 RSSM 状态推进

---

## 3. 总体训练链路

当前训练主线可以概括为：

1. 环境交互产生 transition；
2. replay buffer 存储 transition 与上一时刻 latent；
3. 采样 `(Batch, Time+1)` 的 slice 序列；
4. 进入 Dreamer 主干更新：
   - 编码器
   - Recurrent State-Space Model（RSSM）
   - reward / continuation / actor / value
5. 从 latent 中读出结构视图：
   - Map / Region View（`M_t`）
   - Object / Slot View（`O_t`）
   - Global View（`g_t`）
   - Rule Context（`rho_t`）
6. 用 `feat + action + rho_t` 学习统一动作后果 latent：`z_eff`
7. 将 `z_eff` 解码为结构变化：
   - `delta_map`
   - `delta_obj`
   - `delta_global`
   - `event`
8. 进行对象化训练（Phase 1B）
9. 进行规则原型监督与可执行规则路径训练（Phase 2）
10. 进行 `2/4/7-step rho shadow rollout`
11. 通过 rollout gate、Atari closed-loop、baseline-relative budget 做阶段 sign-off

---

## 4. 当前实现与当前边界

### 4.1 当前已经实现的能力

当前仓库已经实现了以下主链能力：

- `StructuredReadout`
- `EffectModel`
- `EffectHeads`
- `ReachabilityHead`
- `GoalProgressHead`
- `ObjectificationModule`
- `OperatorBank`
- `BindingHead`
- `SignatureHead`
- `RuleUpdateHead`
- `RuleMemory`
- `RuleApply`
- `2-step / 4-step / 7-step rho shadow rollout`
- rollout gate
- Atari closed-loop sign-off
- baseline-relative 预算判定

### 4.2 当前尚未实现的能力

当前仓库尚未实现以下更强能力：

- full structured rollout
- full latent rollout
- planner 主路径
- actor / value 直接消费规则路径
- 将规则路径写回并替代主 RSSM 推进
- 专门的 event-rich / freshness-aware / multi-horizon replay 采样器主线

### 4.3 当前 replay 的准确定位

当前默认主线仍然是 **普通 slice replay**。
仓库中已经有结构感知 priority 计算逻辑，也支持 prioritized replay，但当前默认 Atari Phase2 主线并未启用专门的：

- event-rich replay
- freshness-aware replay
- multi-horizon replay

因此，准确表述应为：

> 当前默认训练主线仍是普通 slice replay；
> 还没有专门的 event-rich / freshness-aware / multi-horizon replay 采样机制。

---

## 5. 核心设计原则

### 5.1 不推翻 Dreamer 主干
当前不是另起一套完全独立的符号系统，而是在 Dreamer 主干之上长出结构接口、对象化模块与规则接口。

### 5.2 先学统一 effect，再学规则
当前不直接将动作拆解到很多独立通道，而是先学统一动作后果 latent：`z_eff`，再从中解码结构变化与规则相关量。

### 5.3 对象化是训练过程，不是假定真值
当前 `Object / Slot View (O_t)` 不是对象真值，而是通过对象化训练逐步变成更可靠的对象承载体。

### 5.4 规则接口先走 shadow path，不直接接 planner
当前先把规则路径做成可检索、可融合、可一跳应用、可多步 `rho` shadow rollout 的模块，再讨论更强推理路径。

---

## 6. 阶段定义（最新版）

## 6.1 Phase 1A：结构化 Dreamer baseline

### 目标
- 保持 `s_t = (h_t, z_t)` 主干稳定训练；
- 从主干 latent 中稳定读出：
  - Map / Region View（`M_t`）
  - Object / Slot View（`O_t`）
  - Global View（`g_t`）
  - Rule Context（`rho_t`）
- 学出统一动作后果表示 `z_eff`
- 学出基础结构变化读出：
  - `delta_map`
  - `delta_obj`
  - `delta_global`
  - `event`
- 提供弱结构辅助读出：
  - weak reachability
  - weak goal-progress

### 当前实现状态
已接入：

- `StructuredReadout`
- `EffectModel`
- `EffectHeads`
- `ReachabilityHead`
- `GoalProgressHead`

### 当前训练内容
Phase 1A 当前主要包含两组损失：

#### 结构读出损失
- `struct_map`
- `struct_obj`
- `struct_global`

#### effect 相关损失
- `delta_map`
- `delta_obj`
- `delta_global`
- `event`

此外，还包括两条弱结构读出头：

- `reach`
- `goal`

### 当前边界
- `M_t / O_t / g_t / rho_t` 仍是从 latent 中读出的结构视图，不是外部硬标签真值；
- `reach / goal` 目前仍是弱结构读出头，不应解读为正式规划器或精确几何可达性求解器；
- 当前 `reach` 更接近“变化强度的弱 proxy”，还不是动作条件下的显式地图可达建模。

### 验收标准
- 主干预测稳定；
- `M_t / O_t` 可追踪且不塌缩；
- `delta_map / delta_obj / delta_global` 可学习；
- `event` 与显式变化有相关性；
- weak reachability / weak goal-progress 对训练过程有稳定辅助作用。

---

## 6.2 Phase 1B：对象化过程训练（Objectification）

### 目标
- 让 `O_t` 从 slot 容器逐步变成更可靠的对象化承载体；
- 提升对象级稳定性、局部性和关系可复用性。

### 当前实现状态
已接入：

- `ObjectificationModule`

已训练三组主损失：

- `L_obj-stable`
- `L_obj-local`
- `L_obj-rel`

已输出对象化诊断量：

- `M_obj`
- `slot_match`
- `slot_concentration`
- `motif_entropy`
- `object_interface`

### 当前边界
- 当前对象化仍然是训练过程，不是显式对象真值系统；
- 当前 `O_t` 更准确地说是“可用对象承载体”，不是已完成的对象本体库；
- 当前多 seed 结果表明：Phase 1B 仍然是 seed 方差的重要来源之一，特别是在 `slot_match` 与 `object_interface` 上。

### 验收标准
- slot matching 稳定性提高；
- effect 更局部化；
- 相似结构在不同轨迹中可复用；
- `M_obj` 达到 Phase 2 进入阈值。

---

## 6.3 Phase 2A：规则原型监督层（Operator / Binding / Signature / Rule Update）

### 前提
- `M_obj` 达到可用阈值；
- 主干与 Phase 1A / 1B 训练稳定。

### 目标
- 将连续 effect 压缩成少数更新算子原型；
- 学出初版 binding 与 signature；
- 学出规则上下文变化 `delta_rule` / `delta_rho`。

### 当前实现状态
已接入：

- `OperatorBank`
- `BindingHead`
- `SignatureHead`
- `RuleUpdateHead`

### 当前输出
- `q_u`：operator posterior
- `q_b`：binding posterior
- `q_sigma`：signature
- `delta_rule`

### 当前边界
- 当前监督仍主要来自 proxy target，不是显式人工规则标签；
- 当前 `BindingHead` 使用固定 5 类：
  - `instance`
  - `type`
  - `relation`
  - `region`
  - `backbone`
- 当前瓶颈更偏向监督与解耦质量，而不是“类别数量不够”。

### 验收标准
- operator 使用分布不塌缩；
- binding 输出不退化成纯噪声；
- signature 对 scope / duration / impact 有初步解释力；
- `delta_rule` 对 `rho` 变化有可学习性。

---

## 6.4 Phase 2B：可执行规则接口（Executable Rule Path）

### 目标
- 将规则原型输出推进成可执行接口，而不是停留在监督头；
- 形成：
  - rule retrieval
  - rule fusion
  - one-step rule apply

### 当前实现状态
已接入：

- `RuleMemory`
- `RuleApply`

当前前向链路已经形成：

`rule_update_head -> rule_memory.retrieve -> rule_apply -> rho_next_pred`

### 当前训练内容
- `memory_delta_rule`
- `delta_rule_fused`
- `rho_next_pred`
- `rule_apply_loss`
- retrieval agreement
- memory usage
- memory fresh usage
- memory write rate
- executable gate

### 当前边界
- 当前已经有 executable rule path，不能再描述为“没有 RuleMemory / retrieval / apply”；
- 但当前执行仍然主要限定在 `rho` 通道上，不等于完整结构世界模型都由规则路径驱动；
- 当前 `RuleMemory` 已经引入 freshness 逻辑，但 retrieval calibration 与 late-stage stability 仍是鲁棒性问题来源之一。

### 验收标准
- retrieval agreement 稳定；
- rule apply error 稳定；
- memory usage / fresh usage 不塌缩；
- executable gate 可通过。

---

## 6.5 Phase 2C：多步 `rho` shadow rollout（2 / 4 / 7-step）

### 目标
- 验证规则接口在多步条件下是否仍可稳定使用；
- 从 one-step rule apply 推进成 multi-step `rho` shadow rollout。

### 当前实现状态
已接入：

- `2-step rollout`
- `4-step rollout`
- `7-step rollout`

已接入：

- two-step rollout gate
- long rollout gate
- curriculum 控制逻辑
- four-step curriculum 的 freshness-aware 启停条件

### 当前真实含义
当前 rollout 是：

> **`rho-only shadow rollout`**

即：

- `rho_t` 可以使用前一步规则预测继续滚动；
- 但 `M_t / O_t / g_t / feat / action` 仍然主要来自 teacher-forced 的真实轨迹路径。

### 这意味着
当前已经不是“只有 one-step apply”，但也还不是：

- full structured rollout
- full latent rollout
- actor / planner 主推理路径

### 验收标准
- two-step rollout 稳定；
- four-step rollout 在 curriculum 下可训练且不破坏主干；
- seven-step 指标不明显爆炸；
- rollout gate 可通过。

---

## 6.6 Phase 2D：Atari closed-loop sign-off（当前主验证阶段）

### 目标
- 不只验证“规则链路存在”，而是验证：
  - executable path 可运行
  - rollout path 可运行
  - task 不退化
  - baseline-relative 预算不过线

### 当前实现状态
已接入：

- `phase2_executable_ready`
- `phase2_rollout_ready`
- `task_ready`
- `baseline_relative_ready`
- `atari_closed_loop`

### 当前意义
这是当前主线的 sign-off 层。它回答的是：

- 当前训练是否通过结构 + rollout + 任务 + 预算闭环

而不是：

- 当前系统已经完成 planner
- 当前系统已经完成 full rule-driven control

### 当前多 seed 现实
多 seed 结果表明：

- 单 seed 成功已不是伪阳性；
- 但跨 seed 方差仍然较大；
- 当前仍处在 “promising candidate, not stable baseline” 阶段。
- 2026-03-12 的 replay rerun 与跨机器复查进一步说明：
  - `prioritized replay` 是当前最强的不稳定放大器；
  - `buffer.prioritized=False` 还不是“最终默认解”，但已经足够作为当前开发安全线；
  - 当前默认主线 `current_head` 仍保留更高 ceiling，因此更适合作为性能参考线，而不是唯一开发基线。

当前已观察到的失败模式至少包括：

#### 失败模式 A：结构消费者偏弱
- retrieval 健康
- 任务得分可高
- 但 `slot_match / object_interface` 不达 baseline-relative 预算

#### 失败模式 B：late-stage freshness / retrieval instability
- 中段规则路径恢复正常
- 后段 `fresh_usage` 与 retrieval agreement 再次下滑
- 任务表现一同退化

### 验收标准
- 多 seed 结果不依赖单 seed 偶然成功；
- executable / rollout / task / baseline-relative 同时满足；
- 能区分：
  - retrieval/freshness 问题
  - objectification/structure robustness 问题
  - 任务性能问题

---

## 6.7 Phase 3：事件链路（proposal -> segment -> buffers）

### 目标
- 将训练样本按时间尺度拆分成中频/长时事件训练对象；
- 后续支持 event-rich / multi-timescale replay。

### 当前状态
- 文档层已定义；
- 当前主线仓库尚未形成完整 event buffer / long-horizon buffer 主训练链。

### 说明
- 这一阶段仍属于后续阶段，不应与已经落地的 Phase 2B / 2C 混淆。

---

## 6.8 Phase 4：cross-game prior bank + consolidation

### 目标
- 沉淀跨游戏可复用规则先验；
- 支持更稳的跨任务迁移与复用。

### 当前状态
- 仍属后续阶段；
- 当前主线尚未进入 prior bank 主训练期。

---

## 6.9 Phase 5：Selective SL + 多层反事实预测

### 目标
- 只在少数高价值歧义节点启用高成本推理预算；
- 将更强规则链路用于 selective reasoning。

### 当前状态
- 仍属后续阶段；
- 当前主线尚未进入 Selective SL 训练期。

---

## 6.10 并行 Phase A：R2-Dreamer 风格前端增强（仅像素输入场景）

### 目标
- 保护 Atari / 渲染版任务中的小目标与局部关键结构；
- 为后端结构化世界模型提供更稳的视觉编码基础。

### 当前定位
- 这是像素输入前端增强并行线；
- 不替代后端的结构化世界模型主线。

---

## 7. 当前最准确的系统定位

当前仓库的最准确定位应为：

> 当前仓库已经不再是“只有 Phase 1A / 1B / 2 监督头”的状态，
> 而是已经具备：
> - 结构读出
> - 对象化训练
> - 规则原型监督
> - RuleMemory / RuleApply 可执行规则接口
> - 2/4/7-step `rho` shadow rollout
> - rollout gate
> - Atari closed-loop sign-off

同时必须保留下面这个限制描述：

> 当前 rollout 仍然是 **`rho-only shadow rollout`**，
> 不是 **full structured rollout**，
> 也不是 **full latent rollout**。

---

## 8. 当前主要问题

当前主线已经不再停留在“模块有没有”，而是进入“鲁棒性与 sign-off 质量”的阶段。当前主要问题包括：

1. **Phase 1B / objectification robustness 不足**
   - 某些 seed 上 `slot_match` 与 `object_interface` 长期偏弱；
2. **late-stage freshness / retrieval stability 不足**
   - 某些 seed 在后期出现 freshness 与 retrieval 共同退化；
3. **当前 `reach` 仍偏弱**
   - 更接近变化强度 proxy，尚未成为动作条件下的地图可达建模；
4. **当前 replay 是最强不稳定放大器**
   - 默认 `prioritized replay` 在多轮 `current_head vs no_prio` 对照中反复放大 deterministic collapse 与 rerun variance；
   - `no_prio` 当前更像开发安全线，`current_head` 更像性能参考线；
   - `prio_mean / prio_lowalpha` 目前都没有成为稳定折中解。
5. **当前 `slots32` 容量线不能转正**
   - `map_slots=32, obj_slots=32` 在 20k 有局部正信号；
   - 但 50k 三机对照显示它整体放大了 `mode_minus_raw` 与 split，不适合作为当前默认配置。
6. **当前 Phase 2 shadow rollout 仍未成为主推理消费者**
   - 还没有反向增强 full structured prediction 或 actor / planner 路径。

---

## 9. 当前最合理的下一步

当前最合理的下一步不是增加 planner，不是回 ARC3，也不是继续横向堆新模块，而是：

1. 继续做 **robustness round**，但主线先收窄到双线结构：
   - 开发安全线：`no_prio`
   - 性能参考线：`current_head`
2. 在 replay 结论进一步收稳之前，暂停：
   - 新 replay knob 扩展
   - `blend0`
   - `slots32`
   - actor 正则 / dynamic schedule / planner
3. 将 sign-off 从“可运行”推进到“更稳的 baseline 候选”
4. 优先把规则路径从 `rho-only shadow rollout` 推到：
   - `rho-conditioned structured prediction consumer`
   - 先让 `delta_rule_fused / rho_next_pred` 帮助 `delta_map / delta_obj / delta_global`
   - 暂不直接进入 actor / planner 主推理路径
   - 第一刀只做 prediction-side residual consumer，保持 replay、planner、actor/value 路径不变
5. 在此基础上，再决定：
   - 是否继续扩更长 rollout
   - 是否进入更强推理路径
   - 是否回 ARC3

### 当前执行口径（2026-03-12 更新）

当前阶段的实验与开发口径固定为：

- repo 默认配置不立即改成 `no_prio`；
- 但所有新功能开发，优先在 `no_prio` 线上验证；
- 同时保留 `current_head` 作为 ceiling / regression reference；
- 所有新功能先做 `20k` smoke A/B，再升 `50k`。

当前冻结项：

- `direct_target_blend = 0.25`
- restored Phase1B robustness
- `match_gate = soft`
- `four_step_curriculum_warmup_updates = 1500`
- tri-mode eval
- eval-state actor diagnostics
- `mode_mix = 0.0`

当前 rule-consumer 的最小实验口径：

- `20k`
- `seed_3/4/5`
- `no_prio` vs `no_prio_rule_consumer`
- 先看 `raw_mode / mode / sample_minus_mode / deterministic collapse`

当前 rule-consumer 版本脉络：

- `v1`
  - 直接对 `delta_map / delta_obj / delta_global` 注入 residual
  - `residual_scale = 0.1`
  - 结论：太激进，常把 `mode_minus_raw` 放大
- `v2`
  - 保持 `map/obj/global` 注入
  - `residual_scale = 0.03`
  - residual 乘 `artifact.gate.detach()`
  - 结论：校准偏移有所收敛，但局部结构仍被改得过深
- `v3`
  - 缩成 `global-only`
  - 保留 `residual_scale = 0.03`
  - 保留 `* artifact.gate.detach()`
  - 结论：平均 deterministic 对齐明显改善，但 `seed_4` clean rerun 仍出现 scheduled-eval 回撤
- `v4`
  - 保持 `global-only`
  - 保持 `residual_scale = 0.03`
  - 保持 `* artifact.gate.detach()`
  - 新增 late enable schedule
  - 目标：验证 `seed_4` 问题是否主要来自 consumer 介入时机过早
- `v5`
  - 保持 `global-only`
  - 保持 `residual_scale = 0.03`
  - 保持 `* artifact.gate.detach()`
  - 保持 late enable
  - 新增 `phase2 gate` threshold enable
  - 目标：验证 `seed_4` 问题是否主要来自 phase2 尚未稳定时 consumer 仍然接管过多

当前升级规则：

- 在 `seed_4` 的 clean `20k` 控制实验不稳定之前，不升 `50k`

---

## 10. 当前代码状态说明（2026-03 最新）

当前仓库已经实现了从 Phase 2A 到 Phase 2C 的连续推进：

- 不仅有 `OperatorBank / BindingHead / SignatureHead / RuleUpdateHead`
- 还已经有 `RuleMemory / RuleApply`
- 以及 `2/4/7-step rho shadow rollout`、对应 gate、Atari closed-loop

因此，后续讨论中不应再把当前仓库描述成：

> “只有 Phase2 监督头，没有 rule retrieval / apply / rollout”

更准确的说法应是：

> 当前仓库已经有 executable rule path，但该路径目前仍限定在 **`rho-only shadow rollout`**，还没有扩展为 **full structured rollout / full latent rollout / planner 主路径**。

---

## 11. 附：术语说明（当前文档内）

### `feat`
Dreamer / RSSM 当前时刻的压缩状态特征，不是原始像素。

### `M_t`
Map / Region View（地图 / 区域视图），更偏空间与区域组织。

### `O_t`
Object / Slot View（对象 / 槽视图），当前阶段仍是训练中的对象承载体，不是对象真值。

### `g_t`
Global View（全局视图），更偏全局状态。

### `rho_t`
Rule Context（规则上下文），表示当前规则/模式上下文，用于调制动作后果。

### `z_eff`
统一动作后果 latent，表示：
在当前 `feat`、当前 `action`、当前 `rho_t` 条件下，本步动作后果的统一压缩表示。

### `delta_rule`
规则上下文变化量。

### `rho_next_pred`
通过 rule path 一步应用后得到的下一时刻规则上下文预测。

### `rho-only shadow rollout`
只在 `rho` 通道上进行多步 rollout；其他结构视图与主 latent 仍然主要走 teacher-forced 的真实路径。

---
