# ARC3 / Atari 统一结构化 Dreamer 项目文档（实施参考版）

版本：v0.9  
日期：2026-03-09  
定位：项目设计文档 / 里程碑文档 / 接口参考文档

---

## 1. 文档目的

本文档用于把当前讨论过的方案收束成一套可执行的项目参考文档，目标不是给出最终理论终局，而是明确：

1. 当前架构的主线是什么。  
2. 哪些模块先做，哪些模块后做。  
3. Dreamer、对象化过程、规则原型、事件链路、跨游戏先验、Selective SL 各自放在哪里。  
4. Atari 与 ARC-AGI-3 如何共用后端、分开前端。  
5. 每个阶段的输入、输出、训练频率、验收标准是什么。

---

## 2. 执行摘要

当前最合理的主线不是“大而全终局系统”，而是：

- 用 **Dreamer 主干** 解决“当前游戏里下一步会怎样”。
- 用 **对象化过程训练** 把 slot 从容器逐步变成可用的对象化结果。
- 用 **operator / binding / signature** 把变化抽象成规则原型。
- 用 **event proposal -> segment -> event buffer -> long-horizon buffer** 把不同时间尺度的训练样本拆开。
- 用 **cross-game prior bank** 慢慢沉淀跨游戏结构先验。
- 用 **Selective SL** 只在少数高价值歧义节点上做反事实推理。

总体设计可压成：

**像素或网格输入 -> 统一世界模型主干 -> 双视图结构读出 -> 统一动作 effect 表示 -> 规则原型层 -> 事件链路 -> 跨游戏先验层 -> Selective SL**

其中：

- 对 **Atari**：前端默认采用 **R2-Dreamer 风格表征增强**，优先保住小对象和局部关键线索。
- 对 **ARC-AGI-3**：直接从原生 grid/state 进入，不需要先走视觉表征增强；但仍然需要对象化过程训练。

---

## 3. 关键设计原则

### 3.1 主线与并行线分离

- **主线**：结构化 Dreamer、对象化过程、规则原型、事件链路、跨游戏先验、Selective SL。  
- **并行线**：像素输入场景下的表征增强（例如 R2-Dreamer 前端）。

R2-Dreamer 应作为 Atari / 渲染版 ARC3 的前端默认基线，但不替代后端的结构建模问题。

### 3.2 分析层次不等于建模模块

“表示确认 / 局部动力学 / 规则确认 / 绑定确认 / 结构发现”这 5 个层次适合分析不确定性，但不能做成 5 套并列动作模型。真实环境中，一次动作可以同时影响多个层次。

### 3.3 动作不是分层模块，动作只激活统一 effect

动作是输入；规则是当前状态下被激活的潜在更新机制；结果是具体 effect realization。  
因此，系统应该学习统一的 action-conditioned effect，而不是为不同层次单独建模。

### 3.4 对象化不是先验真值，而是过程结果

对象化不必一开始就给出精确对象标签。更合理的方式是先给统一 slot 容器，让对象化通过动作—反馈—结构更新逐步浮现，然后将浮现出来的对象化结果显式读出并进入后续模块。

### 3.5 变量按生命周期组织，训练按频率组织

不要先问“这个变量叫什么”，先问：

- 它住在哪一层。  
- 它活多久。  
- 它什么时候解码。  
- 它用多长窗口监督。  
- 它多久更新一次。

---

## 4. 整体架构

### 4.1 两种输入路线

#### Atari 路线

Pixels  ->  R2-style visual encoder  ->  slot/object adapter  ->  Dreamer core

#### ARC-AGI-3 路线

Grid / JSON state  ->  grid/pattern encoder  ->  slot/object adapter  ->  Dreamer core

注意：ARC-AGI-3 的原生输入是结构化 grid/state，而不是“已经对象化”的输入；仍然需要对象化过程训练。

### 4.2 统一后端

两种输入在后端尽量统一成：

- map/grid view
- object/slot view
- global state
- rule context
- effect packet
- event chain
- prior bank
- Selective SL

---

## 5. 核心状态与公式（按顺序）

### 公式 F1：Dreamer 核心状态

`s_t = (h_t, z_t)`

含义：
- `h_t`：循环记忆 / deterministic state
- `z_t`：随机 latent / stochastic state

说明：这一层尽量保持统一，不要把大量高层语义直接塞进核心状态。

### 公式 F2：结构读出状态

`x_t = (M_t, O_t, g_t, ρ_t)`

含义：
- `M_t`：map/grid/region view
- `O_t`：object/slot view
- `g_t`：global state
- `ρ_t`：规则后验压缩表示

### 公式 F3：effect latent

`z_eff_t = E(s_t, a_t, ρ_t)`

含义：
- `z_eff_t` 是统一的动作后果表征。  
- 不直接让主干去背所有细字段，而是先压成 effect latent。

### 公式 F4：effect packet

`c_t = (ΔM_t, ΔO_t, Δg_t, Δρ_t, e_t, ω_t, d_t, κ_t)`

含义：
- `ΔM_t`：地图/区域结构变化
- `ΔO_t`：对象/slot 变化
- `Δg_t`：全局状态变化
- `Δρ_t`：规则后验变化
- `e_t`：事件摘要
- `ω_t`：作用范围 / binding 权重
- `d_t`：持续时间
- `κ_t`：长期影响 / future divergence

说明：这是解释接口，不应被理解成 8 个同级主任务；训练时应先学统一 effect latent，再按需解码这些字段。

### 公式 F5：规则原型

`r_t = (u_t, b_t, σ_t)`

含义：
- `u_t`：update operator，更新算子原型
- `b_t`：binding scope，绑定层级
- `σ_t`：spatiotemporal signature，时空影响签名

### 公式 F6：事件 proposal 分数

`P_evt(t) = α·U_explicit + β·U_residual + γ·((1-M_obj)·U_struct_proxy + M_obj·U_struct) + δ·U_rule`

含义：
- `U_explicit`：显式变化（reward、done、global flags）
- `U_residual`：模型残差/异常更新
- `U_struct_proxy`：弱结构代理变化
- `U_struct`：较成熟的结构变化
- `U_rule`：规则后验变化
- `M_obj`：对象化成熟度，用于控制何时相信结构变化检测

### 公式 F7：Selective SL 触发分数

`S_SL = M_obj · A_task · C_diag · G_future`

含义：
- `M_obj`：当前对象化接口可用性
- `A_task`：任务相关歧义
- `C_diag`：当前动作的诊断能力
- `G_future`：澄清后未来收益

说明：只有在对象化接口、规则后验、影响读出都基本可用时，Selective SL 才应该开启。

---

## 6. 符号与词汇表

| 符号/词汇 | 含义 |
|---|---|
| `s_t` | Dreamer 核心 latent state |
| `h_t` | recurrent / deterministic memory |
| `z_t` | stochastic latent |
| `x_t` | 结构读出状态 |
| `M_t` | map/grid/region view |
| `O_t` | object/slot view |
| `g_t` | global state（inventory、mode、score、life、phase） |
| `ρ_t` | 规则后验压缩表示 |
| `u_t` | 更新算子原型 |
| `b_t` | 绑定层级 |
| `σ_t` | 时空影响签名 |
| `z_eff_t` | effect latent |
| `c_t` | effect packet |
| `e_t` | 事件摘要 token |
| `ω_t` | 作用范围 / binding 权重 |
| `d_t` | 持续时间 |
| `κ_t` | 长期影响 / future divergence |
| `M_obj` | 对象化成熟度 / 对象化接口可用性 |
| `P_evt` | 事件候选分数 |
| `S_SL` | Selective SL 触发分数 |

---

## 7. 变量分层与训练分频

### 7.1 四层变量

#### 层 1：核心状态
- `s_t = (h_t, z_t)`
- 长期滚动，必须小而稳

#### 层 2：结构读出
- `x_t = (M_t, O_t, g_t, ρ_t)`
- 是后续规则、事件、可达性和 effect 的统一接口

#### 层 3：effect 读出
- `c_t = (ΔM_t, ΔO_t, Δg_t, Δρ_t, e_t, ω_t, d_t, κ_t)`
- 是动作后果的解释接口，不是平级主状态字段

#### 层 4：决策分析量
- `A_task, C_diag, G_future, S_SL`
- ambiguity, controllability, future gain, importance 等
- 多数不应常驻 state，而应按需计算、训练时回填、周期性校准

### 7.2 四种训练频率

#### 高频主干
- Dreamer world model
- reward / continue
- `ΔM_t, ΔO_t, Δg_t`

#### 事件驱动
- `e_t`
- `Δρ_t`
- operator posterior
- 初版 binding

#### 长时回填
- `d_t`
- `κ_t`
- reachability / topology change
- signature 相关量

#### 跨游戏周期更新
- operator consolidation
- binding priors
- signature priors
- prototype merge/split

---

## 8. 对象化过程训练：压缩后的 3 组 loss

前面分析过对象化至少涉及 5 类性质：
- 稳定性
- 局部作用性
- 反事实独立性
- 交互可组合性
- 规则复用性

训练时不应把它们拆成 5 个同级 loss。建议压缩成下面 3 组。

### 8.1 `L_obj-stable`

负责：
- 稳定性
- 可追踪性
- 基本一致性

包含但不限于：
- temporal slot consistency
- slot matching consistency
- 局部表示稳定

目标：这些 slot 至少要像“同一个东西”。

### 8.2 `L_obj-local`

负责：
- 局部作用性
- action effect 稀疏性
- 反事实局部可分解性

包含但不限于：
- action-conditioned effect localization
- slot-wise effect sparsity
- counterfactual separability

目标：动作应主要影响少数局部单元，而不是平均扰动整个状态。

### 8.3 `L_obj-rel`

负责：
- 交互可组合性
- 规则复用性
- 相似单元共享 operator 的倾向

包含但不限于：
- interaction motif consistency
- reusable operator assignment tendency
- relation predictability

目标：slot 不仅可追踪，还能成为规则和交互的承载器。

### 8.4 对象化成熟度 `M_obj`

从上述三组训练信号蒸馏出一个对象化成熟度分数 `M_obj`，用于：
- proposal 阶段控制是否相信结构变化信号
- operator/binding/signature 的启用时机
- Selective SL 的门控

---

## 9. 规则原型层：operator / binding / signature

### 9.1 operator bank

目标：把大量具体变化压成少数可复用的更新算子原型。

接口建议：
- 输入：`s_t, a_t, O_t, M_t, ρ_t`
- 输出：`q(u_t | context)`

### 9.2 binding head

目标：判断当前规则更像绑定在：
- instance
- type
- relation
- region
- backbone

接口建议：
- 输入：`u_t` 和局部上下文
- 输出：`q(b_t | u_t, context)`

### 9.3 signature head

目标：输出规则的时空影响签名：
- 作用范围
- 持续时间
- 长期影响

接口建议：
- 输入：`u_t, context`
- 输出：`q(σ_t | u_t, context)`

---

## 10. 事件链路：proposal -> segment -> buffers

### 10.1 统一事件检测原则

事件检测不先做语义判定，而先做：
- 异常更新
- 稀有更新
- 高影响更新

### 10.2 Event Proposal

用途：
- 快速提议“这里可能不是普通平滑动力学”

建议字段：
- `episode_id`
- `t`
- `action`
- `score_total`
- `score_global`
- `score_struct`
- `score_pred_err`
- `score_rule_shift`
- `proposal_type`

### 10.3 Event Segment

用途：
- 将相邻 proposal 聚成一个片段
- 表示 onset / peak / offset

建议字段：
- `segment_id`
- `t_start`
- `t_peak`
- `t_end`
- `source_proposals`
- `effect_signature`
- `delta_M_summary`
- `delta_O_summary`
- `delta_g_summary`
- `delta_rho_summary`
- `omega_est`
- `duration_est`
- `impact_est`

### 10.4 Event Buffer

用途：
- 训练中短期事件与规则更新

建议存储：
- 片段前后上下文
- `event_token_target`
- `delta_rho_target`
- `omega_target`
- `priority`
- `rarity_score`

### 10.5 Long-Horizon Buffer

用途：
- 训练长期变量

建议存储：
- `duration_target`
- `impact_target`
- `reachability_delta_target`
- `value_delta_target`
- `branch_divergence_target`
- `goal_progress_delta`

---

## 11. 分阶段计划（主线 + 并行线）

### Phase 1A：结构化 Dreamer baseline

#### 目标
- `s_t=(h_t,z_t)` 主干稳定
- `x_t=(M_t,O_t,g_t,ρ_t^weak)` 可读
- `z_eff` 可用
- 基础 effect 可读：`ΔM, ΔO, Δg, e`
- weak reachability / weak goal-progress 可读

#### 输入
- Atari：R2-style encoder 输出的 latent features
- ARC3：grid/state encoder 输出的结构 features

#### 输出
- 基础 world model rollout
- 双视图结构读出
- effect latent

#### 不做
- 正式 operator bank
- binding/signature
- event buffers
- cross-game prior
- SL

#### 验收标准
- 主干预测稳定
- `M_t/O_t` 至少可追踪
- `ΔM/ΔO/Δg` 可学习
- weak goal-progress 对目标模式变化有相关性

---

### Phase 1B：对象化过程训练

#### 目标
- 让 `O_t` 从 slot 容器变成对象化结果载体

#### 训练
- `L_obj-stable`
- `L_obj-local`
- `L_obj-rel`

#### 输出
- `M_obj`
- 更可靠的 `O_t`

#### 验收标准
- slot matching 稳定性提高
- effect 更局部化
- 相似结构在不同轨迹中可复用性提升

---

### Phase 2：operator bank + binding/signature 初版

#### 前提
- `M_obj` 达到可用阈值

#### 目标
- effect 被压成少数更新算子原型
- 初步形成 binding 与 signature

#### 训练
- operator posterior
- binding posterior
- signature posterior
- `Δρ`

#### 验收标准
- operator 使用分布不塌缩
- binding 输出不全是噪声
- signature 对 `ω,d,κ` 有初步解释力

---

### Phase 3：event proposal / segment / buffers

#### 目标
- 将训练样本按时间尺度分离

#### 训练
- proposal 生成
- proposal 聚成 segment
- event buffer 训练中频变量
- long-horizon buffer 训练长时变量

#### 验收标准
- proposal 能稳定抓到显式变化与高影响变化
- segment 不是纯碎点
- `d, κ` 的 hindsight 标签可回填

---

### Phase 4：cross-game prior bank + consolidation

#### 目标
- 开始沉淀跨游戏结构先验

#### 训练
- operator consolidation
- binding priors
- signature priors
- prototype merge/split
- Bayesian-style reduction

#### 验收标准
- 新游戏 warm-start 比从零更快
- operator bank 数量可控，不无限膨胀
- 相似规则在跨游戏上可复用

---

### Phase 5：Selective SL + 多层反事实预测

#### 目标
- 只在少数高价值歧义节点上用重推理预算

#### 训练
- ambiguity calibrator
- controllability predictor
- future-gain predictor
- SL gate
- shallow branch evaluation

#### 验收标准
- SL 触发频率低但有效
- 每次触发有可观的信息收益或长期收益提升

---

### 并行 Phase A：R2-Dreamer 前端增强（仅像素输入场景）

#### 适用
- Atari 原始像素
- 渲染版 ARC3

#### 目标
- 提升小对象与局部关键线索保真度
- 改善 `M_t/O_t` 稳定性和 `z_eff` 可预测性

#### 对照实验
- baseline encoder
- R2-style encoder / representation objective

#### 验收标准
- 小目标不易丢失
- `O_t` 可追踪性更好
- event / operator 读出更稳定

---

## 12. Atari 与 ARC3 的兼容策略

### Atari
- 前端：R2-Dreamer 风格表征增强作为默认强基线
- 中层：生成 `M_t + O_t`
- 后端：与 ARC3 共享 effect / rule / event / prior / SL 体系

### ARC-AGI-3
- 前端：直接从 grid/state 进入
- 中层：同样生成 `M_t + O_t`
- 后端：共享同一套结构化 Dreamer 后端

### 关键取舍
- ARC3 不需要先视觉增强，但仍然需要对象化过程训练
- Atari 通常更需要前端表征增强，否则小对象丢失会污染后续全部模块

### 执行备注（2026-03-09）
- 长期目标仍是 Atari / ARC3 共享后端、分开前端。
- 但当前实现阶段不再并行调两条训练线，而是先冻结 ARC3、专注 Atari。
- ARC3 冻结点：
  - `ee411c1 fix: wire arc3 special token config`
  - `aa2c1fa refactor: align structured state with masked views`
- 原因：
  - 双环境并行会同时引入前端差异、奖励分布差异、对象化难度差异，难以判断训练改动是否真正有效。
  - Atari 像素线更适合先做单域对照，验证 Phase 1B/2 的训练稳定性与收益。
- 当前阶段的工程原则：
  - ARC3 只保留接口正确性和 smoke 可运行性。
  - 所有新的训练稳定性改动，先在 Atari 基线验证，再择优回灌 ARC3。

---

## 13. 训练与验证方法

### 13.1 单游戏内验证
- 主干 sample efficiency
- event prediction 质量
- `Δρ` 收缩效果
- weak reachability / goal-progress 是否有用

### 13.2 跨游戏泛化验证
- 使用 prior bank warm-start 的适应速度
- operator/binding/signature 是否复用
- 是否减少探索量

### 13.3 事件链路验证
- proposal 命中率
- segment 质量
- long-horizon target 的稳定性

### 13.4 SL 验证
- 触发频率
- 信息收益
- 长期价值提升

---

## 14. 当前最重要的取舍

1. 不先做“大而全终局系统”。  
2. 不把对象化当成硬对象检测。  
3. 不把 map/object 双视图做成过重图系统。  
4. 不把所有分析量都变成状态字段。  
5. 不让 SL 和复杂前端表征先接管主线。  
6. 先把主干、对象化过程、规则原型、事件时间尺度拆分打稳。

---

## 15. 外部启发（非直接照搬）

- DreamerV3：统一 world model 主干、imagined rollout、跨域固定配置。  
- R2-Dreamer：像素场景下的小目标/局部关键信号表征增强。  
- Sophisticated Inference / Sophisticated Learning：高价值不确定性的反事实推理。  
- AXIOM：对象中心先验、在线扩展结构、Bayesian model reduction。  

---

## 16. 近期落地建议（执行版）

### 近期 1
先固定接口，不再继续加字段：
- `s_t=(h_t,z_t)`
- `x_t=(M_t,O_t,g_t,ρ_t^weak)`
- `z_eff`
- `ΔM, ΔO, Δg, e`
- weak reachability / weak goal-progress

### 近期 2
先把对象化过程训练压成 3 组 loss：
- `L_obj-stable`
- `L_obj-local`
- `L_obj-rel`
- 输出 `M_obj`

### 近期 3
再接 operator bank + binding/signature 初版，不做过多扩展。

### 近期 4
再接 proposal -> segment -> event buffer -> long-horizon buffer。

### 近期 5
最后再做跨游戏 prior bank 与 Selective SL。

---

## 17. 本文档的定位说明

本文档不是最终理论终局，而是当前最适合作为项目实现主线的参考版本。后续可以在不破坏主线的前提下，逐步升级到：

- typed object-centric attributed relational hypergraph
- action-conditioned schema/state graph
- typed operators / constraints / programs

但这些更适合作为 **Phase 4 以后** 的升级方向，而不是当前第一阶段的主实现接口。

---

## 18. 里程碑表（执行版）

| 里程碑 | 目标 | 核心交付 | 依赖 | 验收标准 |
|---|---|---|---|---|
| M0 | 实验基础设施 | 统一训练脚手架、日志、评测脚本、replay schema | 无 | 单游戏可稳定跑通、日志字段完整 |
| M1 | Phase 1A 完成 | `s_t`、`M_t/O_t/g_t/ρ^weak`、`z_eff`、`ΔM/ΔO/Δg/e`、弱可达性与弱 goal-progress | M0 | 单游戏 rollout 稳定，基础 effect 可读 |
| M2 | Phase 1B 完成 | `L_obj-stable`、`L_obj-local`、`L_obj-rel`、`M_obj` | M1 | slot 稳定性提升，局部 effect 更稀疏，结构接口可用 |
| M3 | 规则原型层初版 | operator bank、binding 初版、signature 初版、`Δρ` | M2 | operator 不塌缩，binding/signature 不全是噪声 |
| M4 | 时间尺度训练拆分 | proposal / segment / event buffer / long-horizon buffer | M3 | 事件片段质量可接受，`d/κ` 可回填 |
| M5 | 跨游戏先验层 | prior bank、prototype merge/split、consolidation | M4 | 新游戏 warm-start 明显快于从零训练 |
| M6 | Selective SL 初版 | ambiguity/calibrator、controllability、future gain、gate、浅层分支评估 | M5 | 低频触发但有信息收益或长期价值提升 |
| M7 | 像素前端增强并入 | Atari/R2 前端正式并入主线 | M1（并行起步），M3 后正式接入 | 小对象保真与后端稳定性均优于 baseline |

---

## 19. 接口落实清单（第一版）

### 19.1 核心状态接口

```text
s_t:
  h_t: recurrent / deterministic memory
  z_t: stochastic latent
```

### 19.2 结构读出接口

```text
x_t:
  M_t: map/grid/region view
  O_t: object/slot view
  g_t: global state
  rho_t: weak rule context or posterior summary
```

### 19.3 effect 接口

```text
z_eff_t = E(s_t, a_t, rho_t)

c_t:
  delta_M_t
  delta_O_t
  delta_g_t
  delta_rho_t
  e_t
  omega_t
  d_t
  kappa_t
```

### 19.4 规则原型接口

```text
r_t:
  u_t: operator prototype
  b_t: binding scope
  sigma_t: spatiotemporal signature
```

### 19.5 对象化成熟度接口

```text
M_obj_t:
  objectification maturity / structure usability score
```

### 19.6 事件 proposal 接口

```text
proposal_t:
  explicit_score
  residual_score
  struct_proxy_score
  struct_score
  rule_score
  total_score
```

### 19.7 缓冲区接口

```text
StepBuffer:
  raw transition + cheap derived signals

EventBuffer:
  event segment + short horizon context + delta_rho / omega targets

LongHorizonBuffer:
  long window + duration / impact / reachability / value targets

PriorBank:
  operator priors
  binding priors
  signature priors
  diagnostic action priors
```

### 19.8 Selective SL 接口

```text
A_task: task-relevant ambiguity
C_diag: diagnostic controllability
G_future: future gain after disambiguation
S_SL = M_obj * A_task * C_diag * G_future
```

---

## 20. 建议的代码目录骨架（参考）

```text
project/
  encoders/
    atari_r2_frontend.py
    arc3_grid_encoder.py
  world_model/
    rssm_core.py
    struct_readout.py
    effect_model.py
  objectification/
    slot_adapter.py
    obj_losses.py
    obj_maturity.py
  rules/
    operator_bank.py
    binding_head.py
    signature_head.py
    rule_posterior.py
  events/
    proposal.py
    segment_builder.py
    event_buffer.py
    long_horizon_buffer.py
  priors/
    prior_bank.py
    consolidation.py
  sl/
    ambiguity.py
    controllability.py
    future_gain.py
    sl_gate.py
    shallow_branch_eval.py
  training/
    phase1a.py
    phase1b.py
    phase2.py
    phase3.py
    phase4.py
    phase5.py
  eval/
    single_game_eval.py
    cross_game_eval.py
    event_eval.py
    sl_eval.py
```
