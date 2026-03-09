# ARC3 / Atari Structured Dreamer — Phase 1A / 1B / 2 实施参考文档

版本：v1.0  
日期：2026-03-09  
定位：里程碑设计 / 接口落实 / 训练流程 / 测试验收 / Codex 交付参考

---

## 1. 文档目的

本文档是对上一版总设计文档的继续落实，目标是把 **Phase 1A / 1B / 2** 变成可以交给 Codex 实现和逐阶段验证的工程文档。本文档重点回答：

1. 这三个阶段各自解决什么问题，为什么按这个顺序做。  
2. 模块接口如何定义，最少需要哪些输入、输出和缓存。  
3. 训练的 loss 如何组织，哪些必须先做，哪些必须后置。  
4. batch 应该如何组织，哪些数据来自普通 replay，哪些来自事件或长时窗口。  
5. 代码脚本和配置入口如何接到当前代码库。  
6. 每一阶段如何判断“实现到位”，什么标准通过后才能进入下一阶段。  
7. Atari 与 ARC3 如何在训练与验证上共用主干、分开前端。  

本文档默认：

- 对 **Atari 像素输入**，采用 **R2-Dreamer 风格前端**作为默认视觉基线。  
- 对 **ARC3 原生 grid/state**，直接走 grid/state 编码路径，不额外要求视觉前端。  
- 主干仍然建立在当前 `r2dreamer` 的训练骨架之上：`train.py` 负责 Hydra 入口、实例化 `Buffer`、环境、`Dreamer` 和 `OnlineTrainer`；`dreamer.py` 负责 encoder、RSSM、reward/cont、actor/value 及 `model.rep_loss` 选择。  

---

## 2. 对当前设计的再收束

### 2.1 当前真正要做的主线

当前不是做“大而全终局系统”，而是做下面这条主线：

**结构化 Dreamer 主干**  
→ **对象化过程训练**  
→ **operator / binding / signature 初版**  
→ **事件时间尺度拆分**  
→ **跨游戏结构先验**  
→ **Selective SL（后置）**

其中：

- **Dreamer 主干**解决“当前游戏里下一步会怎样”。
- **对象化过程训练**解决“slot 容器如何逐步变成可用的对象化结构单元”。
- **operator / binding / signature** 解决“变化属于哪类更新机制、作用在哪层、时空影响如何”。
- **事件链路** 解决“不同时间尺度的训练样本不能混在同一个 replay 里”。
- **跨游戏先验** 解决“规则原型怎样沉淀，而不是每个游戏从零学”。
- **Selective SL** 只在后面做少数高价值歧义节点上的重型认知增强。

### 2.2 当前最重要的统一原则

#### 原则 A：动作不是分层模块，动作只激活统一 effect

动作不“属于某一层”；动作在当前状态里激活一个潜在更新机制，模型要学的是这个机制产生的统一 effect packet。

可写成：

$$
(s_t, a_t) \rightarrow r_t=(u_t,b_t,\sigma_t) \rightarrow z_t^{eff} \rightarrow c_t \rightarrow s_{t+1}
$$

#### 原则 B：对象化和像素 / 网格是并列视图，不是替代关系

对于后端来说，更合理的统一结构是：

$$
x_t=(M_t, O_t, g_t, \rho_t)
$$

其中：

- $M_t$：map / grid / region view  
- $O_t$：object / slot view  
- $g_t$：global state  
- $\rho_t$：rule posterior / context

对 Atari：`pixels -> visual front-end -> M_t + O_t`  
对 ARC3：`grid/state -> grid encoder -> M_t + O_t`

#### 原则 C：变量按生命周期组织，训练按频率组织

不要先问“变量叫什么”，先问：

- 它住哪一层。  
- 它活多久。  
- 它什么时候解码。  
- 它用多长窗口监督。  
- 它多久更新一次。  

---

## 3. 当前代码库锚点与接口假设

### 3.1 R2-Dreamer 代码骨架（作为主干锚点）

当前公开的 `r2dreamer` 代码骨架已经有以下事实：

- `train.py` 是 Hydra 入口，负责：seed、logdir、`Buffer`、`make_envs`、`Dreamer`、`OnlineTrainer.begin(agent)`。  
- `dreamer.py` 中的 `Dreamer` 已经包含：`encoder`、`rssm`、`reward`、`cont`、`actor`、`value`，并通过 `model.rep_loss` 在 `dreamer | r2dreamer | infonce | dreamerpro` 之间切换。  
- 当前仓库已经支持图像与状态 benchmark，并包含 Atari 100k、DMC、Crafter、Memory Maze 等环境基线。

因此，本方案默认：

- **不推翻现有训练骨架**。  
- **优先在 `Dreamer` 类内部和其邻近模块扩展结构读出 / effect / operator bank**。  
- **配置入口继续沿用 Hydra + `train.py`**。  

### 3.2 ARC3 接口锚点

ARC-AGI-3 官方文档提供：

- 标准化 agent 运行入口，例如 `uv run main.py --agent=random --game=ls20`。  
- 本地 toolkit / API 统一动作空间和 `env.step(...)` 接口。  

因此，本方案默认 ARC3 侧需要做的是：

- 一个 `grid/state -> encoder -> M_t + O_t` 的输入适配器。  
- 一个与 Dreamer 风格 trainer 对接的环境包装层。  

### 3.3 对你的最新分支代码的说明

本文档**没有直接依赖你的私有或最新分支内容**做精确行级映射；当前实现计划主要锚定到：

- `NM512/r2dreamer` 的公开训练骨架。  
- ARC-AGI-3 的公开运行与 toolkit 接口。  

因此，本文档给出的“文件建议”和“接口落点”是**面向当前公开骨架的最小侵入式改造方案**。落到你的最新分支时，Codex 需要先做一次仓库映射，把同名模块或同功能模块对应起来。

---

## 4. 核心状态、读出、effect 与规则表示

### 4.1 核心状态（不变）

$$
s_t=(h_t,z_t)
$$

- $h_t$：循环记忆 / deterministic memory  
- $z_t$：随机 latent / stochastic latent

要求：

- Phase 1A 不大改这层定义。  
- 不要把大量高层字段塞进核心状态。  

### 4.2 结构读出

$$
x_t=(M_t,O_t,g_t,\rho_t)
$$

#### `M_t`
地图 / 网格 / 区域视图，包含：
- 局部 pattern 表征  
- 区域连续性或弱区域组织  
- passability / reachability 的弱读出承载器

#### `O_t`
对象 / slot 视图，初期是**slot 容器**，不是对象真值。  
后续通过对象化过程训练逐步形成对象化结果。

#### `g_t`
全局状态：
- inventory  
- mode / phase  
- score / life  
- 其他全局 flags

#### `ρ_t`
规则上下文或规则后验的压缩表示。  
Phase 1A 先弱读出，Phase 2 再升级成 operator / binding / signature。

### 4.3 effect latent 与 effect packet

先学统一 effect latent：

$$
z_t^{eff}=E(s_t,a_t,\rho_t)
$$

再按需解码为：

$$
c_t=(\Delta M_t,\Delta O_t,\Delta g_t,\Delta \rho_t,e_t,\omega_t,d_t,\kappa_t)
$$

其中：
- $\Delta M_t$：地图 / 区域结构变化  
- $\Delta O_t$：对象 / slot 变化  
- $\Delta g_t$：全局状态变化  
- $\Delta \rho_t$：规则后验变化  
- $e_t$：事件摘要  
- $\omega_t$：作用范围 / binding weights  
- $d_t$：持续时间  
- $\kappa_t$：长期影响 / future divergence

**重要**：训练时不把这些视为 8 个同级主任务，而是视为 `z_eff` 的按需读出。

### 4.4 规则原型表示（Phase 2）

$$
r_t=(u_t,b_t,\sigma_t)
$$

- $u_t$：operator prototype / 更新算子原型  
- $b_t$：binding scope / 绑定层级  
- $\sigma_t$：spatiotemporal signature / 时空影响签名

---

## 5. Phase 1A：结构化 Dreamer baseline

## 5.1 目标

Phase 1A 的目标不是高层规则，也不是跨游戏先验，而是先把以下 5 件事打稳：

1. Dreamer 主干可稳定训练。  
2. `M_t + O_t + g_t + ρ_weak` 能从核心 latent 中稳定读出。  
3. `z_eff` 能承载动作后果。  
4. 基础 effect：`ΔM, ΔO, Δg, e` 能学习。  
5. weak reachability / weak goal-progress 可读。  

## 5.2 模块接口

### 5.2.1 模块列表

| 模块 | 建议文件 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| Core Dreamer | `dreamer.py`（扩展） | obs, prev state | `s_t=(h_t,z_t)` | 保持现有主干 |
| Structured Readout | `structured_readout.py` | `s_t` | `M_t,O_t,g_t,ρ_weak` | 双视图 + 全局 + 弱规则上下文 |
| Effect Model | `effect_model.py` | `s_t,a_t,ρ_weak` | `z_eff` | 统一动作后果 latent |
| Effect Heads | `effect_heads.py` | `z_eff` | `ΔM,ΔO,Δg,e` | 第一阶段只训核心 effect |
| Reachability Head | `reachability_head.py` | `s_t` or `M_t` | weak reachability | 弱可达性读出 |
| Goal Progress Head | `goal_progress_head.py` | `s_t,M_t,g_t` | weak goal-progress | 对 ARC3 很重要 |

### 5.2.2 接口建议（伪代码）

```python
class StructuredReadout(nn.Module):
    def forward(self, state: TensorDict) -> dict:
        return {
            "map_view": M_t,      # [B, T?, N_map, Dm] or [B, T?, H, W, Cm]
            "obj_view": O_t,      # [B, T?, N_slot, Do]
            "global_view": g_t,   # [B, T?, Dg]
            "rule_ctx": rho_weak  # [B, T?, Dr]
        }
```

```python
class EffectModel(nn.Module):
    def forward(self, state, action, rule_ctx) -> Tensor:
        # returns z_eff
        ...
```

```python
class EffectHeads(nn.Module):
    def forward(self, z_eff) -> dict:
        return {
            "delta_map": delta_M,
            "delta_obj": delta_O,
            "delta_global": delta_g,
            "event_logits": e_logits,
        }
```

### 5.2.3 `M_t` 与 `O_t` 的最低要求

#### `M_t`
第一阶段不要求图网络化，只要求能承载：
- 网格局部 pattern  
- 区域弱组织  
- future reachability 的弱线索

#### `O_t`
第一阶段也不要求对象真值，只要求：
- 统一 slot 容器  
- 基本可追踪  
- 后续能被动作 effect 局部影响  

## 5.3 Phase 1A 的 loss 设计

### 5.3.1 主干损失

保持 Dreamer 主干已有损失：

$$
L_{wm} + L_{reward} + L_{cont}
$$

其中：
- $L_{wm}$：世界模型 / RSSM / 表征主损失（Atari 像素路径可直接沿用 R2-Dreamer 表征损失）  
- $L_{reward}$：reward 预测  
- $L_{cont}$：continue / termination 预测

### 5.3.2 第一阶段新增损失

$$
L_{phase1A} =
\lambda_M L_{\Delta M}
+\lambda_O L_{\Delta O}
+\lambda_g L_{\Delta g}
+\lambda_e L_e
+\lambda_R L_{reach}
+\lambda_G L_{goal}
$$

#### `L_ΔM`
地图 / 区域变化预测损失。  
可根据表示形式选：
- MSE / Huber（连续 map features）  
- BCE / CE（离散 grid tokens 或二值局部 mask）

#### `L_ΔO`
对象 / slot 变化预测损失。  
重点不是语义名称，而是：
- 哪些 slot 变了  
- 变化方向和幅度  
- 变化是否局部

#### `L_Δg`
全局状态变化预测损失。  
适用于：inventory、score、life、mode、phase 等。

#### `L_e`
基础事件预测损失。  
第一阶段不追求细语义，只要求让模型能区分“普通平滑动态 vs 有意义后果”。事件 token 可以先从少量显式事件 + 自动聚类的粗事件 code 开始。

#### `L_reach`
弱可达性损失。  
第一阶段不要求完整拓扑图，只要做到：
- 当前局部通达性变化能被感知  
- 某动作是否使局部区域更可达

#### `L_goal`
弱目标推进损失。  
尤其是 ARC3，重点不是直接输出答案，而是通过动作把状态推进到目标模式。因此需要一个弱目标推进头来读：
- 这一步之后是否更接近目标模式  
- 这次 effect 是否推动结构达成

### 5.3.3 Phase 1A 不做什么

- 不训 operator bank  
- 不训 binding/signature  
- 不训 `Δρ` 的强版本  
- 不训 `ω,d,κ`  
- 不开 SL gate

## 5.4 batch 组织

### 5.4.1 batch 类型

Phase 1A 只需要两类 batch：

#### A. 普通 replay sequence batch
从普通 replay 采样固定长度序列：

```python
batch = {
    "obs": [B, T, ...],
    "action": [B, T, A],
    "reward": [B, T],
    "is_first": [B, T],
    "is_terminal": [B, T],
}
```

建议：
- `T_core = 16` 或 32  
- Atari 与 ARC3 都先统一成 sequence batch

#### B. 局部 reachability / goal-progress 辅助 batch
若环境允许，可以从主 batch 中派生，不需要单独 buffer：
- 局部 reachable mask 代理  
- 当前 progress proxy

### 5.4.2 采样原则

- 多游戏训练时，先做 **game-balanced sampling**，不要让某一游戏占绝对多数。  
- Atari 和 ARC3 若混训，先使用 **domain-balanced sampling**：每个 batch 固定比例来自 Atari / ARC3。  
- Phase 1A 先不做 event-balanced sampling。

## 5.5 训练脚本入口建议

### 5.5.1 基于当前 `r2dreamer` 的最小侵入方案

继续用：

```bash
python3 train.py ...
```

新增 Hydra config groups：

```text
configs/
  exp/
    phase1a_atari.yaml
    phase1a_arc3.yaml
    phase1a_mix.yaml
  model/
    structured_readout.yaml
    effect_model.yaml
  env/
    arc3_grid.yaml
    atari_100k.yaml
```

### 5.5.2 推荐配置开关

```yaml
model:
  rep_loss: r2dreamer           # Atari 默认
  use_structured_readout: true
  use_effect_model: true
  use_goal_progress_head: true
  use_reachability_head: true
  use_objectification: false    # Phase 1A 先 false，1B 再 true
  use_operator_bank: false
  use_event_pipeline: false
  use_sl_gate: false
```

### 5.5.3 推荐新脚本

```text
scripts/
  train_phase1a_atari.sh
  train_phase1a_arc3.sh
  train_phase1a_mix.sh
  eval_phase1a.sh
```

## 5.6 Phase 1A 验收标准（进入 1B 前必须通过）

### 单元测试
- 形状测试：`s_t -> x_t -> z_eff -> ΔM/ΔO/Δg/e` 全链路 shape 正确。  
- serialization：checkpoint 存取无误。  
- mixed precision / compile 开关下无类型错误。  
- ARC3 / Atari 前端都能完成一次完整前向与一次更新。

### 训练稳定性
- 至少 3 个 seed、每个 seed 的短跑（例如 20k–50k updates）无 NaN / Inf。  
- loss 曲线无持续爆炸。  
- `ΔM/ΔO/Δg` 比 trivial baseline（no-change / copy-forward）更优。  
- weak reachability / weak goal-progress 至少在 toy / micro-suite 上有效。

### 结构读出可用性
- slot 在相邻时间步的对应关系不完全随机。  
- 局部 effect 能映射到少量 slot / map 区域，而不是平均扰动全状态。  
- Atari 小目标任务上，R2 前端明显优于纯普通 encoder baseline（至少定性或代理指标更稳）。

### 通过标准
Phase 1A 只有在以下三项同时满足时才算通过：
1. 主干稳定训练。  
2. 双视图读出可用。  
3. effect 基础读出（`ΔM/ΔO/Δg/e`）优于简单 baseline。  

---

## 6. Phase 1B：对象化过程训练

## 6.1 目标

Phase 1B 的目标是：让 `O_t` 从“slot 容器”变成“可用的对象化结果载体”。

这里的对象化不是对象真值监督，而是通过训练让某些 slot 逐渐表现出：
- 稳定可追踪  
- 局部可作用  
- 反事实可分离  
- 可形成稳定交互模式  
- 可被后续规则原型复用

## 6.2 为什么不能拆成太多 loss

前面对话里提出过 5 类对象化性质：
- 稳定性  
- 局部作用性  
- 反事实独立性  
- 交互可组合性  
- 规则复用性

分析上很好，但训练时不应该做成 5 个平级损失，否则会再次变量爆炸。  
因此压缩成 3 组 loss：

### 组 A：`L_obj-stable`
负责：稳定性 + 可追踪性

### 组 B：`L_obj-local`
负责：局部作用性 + 反事实局部可分解性

### 组 C：`L_obj-rel`
负责：交互可组合性 + 规则复用倾向

## 6.3 Phase 1B 模块接口

### 6.3.1 新增模块

| 模块 | 建议文件 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| Objectification Module | `objectification.py` | `O_t`, `z_eff`, action, next readouts | object losses + `M_obj` | 对象化过程训练模块 |
| Slot Matching Utils | `slot_matching.py` | `O_t`, `O_{t+1}` | alignment / match | 时间一致性与追踪 |
| Counterfactual Locality Head | `cf_locality.py` | `O_t`, action, z_eff | locality metrics | 支持对象化局部性训练 |

### 6.3.2 输出

```python
objectification_out = {
    "loss_obj_stable": ...,
    "loss_obj_local": ...,
    "loss_obj_rel": ...,
    "objectness_score": M_obj,
    "slot_type_hints": ...  # optional, not hard labels
}
```

## 6.4 Phase 1B 的 loss 设计

### 6.4.1 `L_obj-stable`

目标：slot 在时间上具有一致性和追踪性。

建议包括：
- slot temporal consistency  
- slot matching consistency  
- slot feature smoothness（非事件区间）

公式写成：

$$
L_{obj-stable}=\lambda_{match}L_{match}+\lambda_{temp}L_{temp}+\lambda_{smooth}L_{smooth}
$$

### 6.4.2 `L_obj-local`

目标：动作 effect 更像局部作用，而不是把所有 slot 平均扰动。

建议包括：
- action-conditioned effect sparsity  
- slot-wise effect concentration  
- counterfactual separability proxy

公式写成：

$$
L_{obj-local}=\lambda_{sparse}L_{sparse}+\lambda_{conc}L_{conc}+\lambda_{cf}L_{cf}
$$

### 6.4.3 `L_obj-rel`

目标：slot 之间形成稳定交互模式，并具有规则复用倾向。

建议包括：
- pairwise interaction consistency  
- reusable effect motif consistency  
- operator reuse tendency proxy

公式写成：

$$
L_{obj-rel}=\lambda_{pair}L_{pair}+\lambda_{motif}L_{motif}+\lambda_{reuse}L_{reuse}
$$

### 6.4.4 `M_obj`：对象化成熟度

把三组信号蒸馏成一个对象化成熟度分数：

$$
M_{obj}=f(L_{obj-stable},L_{obj-local},L_{obj-rel},\text{running stats})
$$

`M_obj` 不是训练目标本身，而是：
- event proposal 中结构变化权重的门控  
- operator bank 是否启动的前提  
- SL 是否允许重型分支评估的前提之一

## 6.5 batch 组织

### 仍以 Phase 1A 的 replay sequence batch 为主

Phase 1B 不需要新的 buffer，只需要在 Phase 1A 的 batch 基础上增加：
- slot matching 计算  
- action effect 局部性统计  
- 简单的局部 counterfactual proxy

### 推荐增加一个辅助采样策略

优先采样：
- 有局部显著变化的 transition  
- 有重复交互模式的片段  
- 有对象出现 / 消失 / 推动 / 接触的片段

但此时还不进入正式 event pipeline。

## 6.6 Phase 1B 验收标准（进入 Phase 2 前必须通过）

### 对象化过程指标
- slot matching 显著优于随机对应 baseline。  
- 动作后 effect 在 slot 上具有明显的局部集中性。  
- 相似交互片段中，相似 slot 模式可重复出现。  
- `M_obj` 不再接近随机噪声，且随训练有稳定上升趋势。

### 工程稳定性
- 引入对象化过程后，主干 world model 无明显退化。  
- `ΔM/ΔO/Δg/e` 性能不因对象化 loss 崩溃。  
- 训练速度可接受（建议相对 Phase 1A slowdown 不超过可接受工程阈值，由项目实际机器决定）。

### 通过标准
只有在以下两条同时满足时，Phase 1B 通过：
1. `O_t` 不再只是纯容器，而是具备可用对象化性质。  
2. 主干 world model 与 effect 读出没有被对象化训练拖坏。  

---

## 7. Phase 2：operator bank + binding/signature 初版

## 7.1 目标

这一步的目标不是做全量规则系统，而是先把已有 effect 模式压缩成少数可复用的更新机制：

- operator prototype  
- binding posterior  
- signature posterior

也就是开始回答：
- 这类变化属于哪类更新算子  
- 它更绑定在 instance / type / relation / region / backbone 哪层  
- 它的时空影响更像短时/局部还是长时/结构级

## 7.2 前提条件

Phase 2 必须建立在：
- Phase 1A 已通过  
- Phase 1B 已通过  
- `M_obj` 达到“结构事件与规则学习可用”的下限

## 7.3 模块接口

### 新增模块

| 模块 | 建议文件 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| Operator Bank | `operator_bank.py` | `s_t,a_t,O_t,M_t,g_t` | `q_u` | operator posterior |
| Binding Head | `binding_head.py` | `u_t, context` | `q_b` | binding posterior |
| Signature Head | `signature_head.py` | `u_t, context` | `q_sigma` | 时空签名后验 |
| Rule Update Head | `rule_update.py` | `z_eff, q_u, q_b, q_sigma` | `Δρ` | 规则后验变化 |

### 接口建议（伪代码）

```python
class OperatorBank(nn.Module):
    def forward(self, state, action, map_view, obj_view, global_view):
        return {
            "q_u": q_u,              # operator posterior
            "operator_embed": u_emb,
        }
```

```python
class BindingHead(nn.Module):
    def forward(self, operator_embed, context):
        return {"q_b": q_b}
```

```python
class SignatureHead(nn.Module):
    def forward(self, operator_embed, context):
        return {"q_sigma": q_sigma}
```

## 7.4 Phase 2 的 loss 设计

### 7.4.1 operator loss

目标：把 effect 模式压缩成可复用更新算子。

$$
L_{op}=L_{assign}+L_{proto-consistency}+L_{reuse}
$$

其中：
- `L_assign`：当前 effect 与某些 operator prototype 的匹配  
- `L_proto-consistency`：相似 effect 落到相似 operator  
- `L_reuse`：避免每个样本都独占 operator

### 7.4.2 binding loss

目标：形成作用层级的软后验。

$$
L_{bind}=L_{bind-ce}+L_{bind-consistency}
$$

这里不要求人工真值标签，而可从：
- 同类对象跨实例一致性  
- 区域内外复用差异  
- 关系条件一致性  
- reachability 变化范围

中派生 proxy target。

### 7.4.3 signature loss

目标：形成时空影响签名。

$$
L_{sig}=\lambda_\omega L_\omega + \lambda_d L_d + \lambda_\kappa L_\kappa
$$

但在 Phase 2 中：
- `ω` 可以先做弱版  
- `d, κ` 只做初版 proxy，不做强监督全量版

### 7.4.4 rule update loss

$$
L_{\Delta \rho}=L_{rho-update}
$$

使规则后验变化与实际 effect / event 更一致。

### 7.4.5 总损失

$$
L_{phase2}=L_{op}+L_{bind}+L_{sig}+L_{\Delta \rho}
$$

并与 Phase 1A / 1B 的主损失共同训练，但权重要后置、较轻，避免拖坏主干。

## 7.5 batch 组织

Phase 2 仍以 replay sequence batch 为主，但增加一种**effect-centric 重采样**：

- 优先采样有明显 `ΔM / ΔO / e` 的片段  
- 优先采样对象化成熟度高的样本  
- 对重复模式进行 balanced sampling，避免 operator 全被频繁普通运动占满

此时仍未正式启用 event buffer / long-horizon buffer；那是 Phase 3 的工作。

## 7.6 脚本与配置入口建议

### Hydra 开关

```yaml
model:
  use_structured_readout: true
  use_effect_model: true
  use_objectification: true
  use_operator_bank: true
  use_binding_head: true
  use_signature_head: true
  use_event_pipeline: false
  use_sl_gate: false
```

### 新增脚本

```text
scripts/
  train_phase2_atari.sh
  train_phase2_arc3.sh
  train_phase2_mix.sh
  eval_phase2.sh
```

## 7.7 Phase 2 验收标准（进入 Phase 3 前必须通过）

### 表示层
- operator posterior 不是完全塌缩为单类。  
- binding posterior 不是纯噪声，能体现基本层级差异。  
- signature posterior 至少能区分短时/长时、局部/大范围的粗差异。  

### 与主干兼容性
- Phase 2 训练后，主干 world model 和对象化过程不显著退化。  
- `ΔM/ΔO/Δg/e` 预测能力保持或提升。  

### 可解释性 / 复用性
- 相似 effect 片段倾向于落到相似 operator。  
- 不同 scope 的 effect 在 binding 上有可区分模式。  
- `M_obj` 高的样本上，规则读出比 `M_obj` 低的样本明显更稳定。

### 通过标准
Phase 2 通过的最低要求：
1. operator / binding / signature 初版可用且非塌缩。  
2. 能作为后续 event proposal、event buffer 和 long-horizon buffer 的前置输入。  

---

## 8. 从 Phase 2 进入 Phase 3 的条件

只有当以下条件同时成立时，才进入事件时间尺度拆分：

1. `ΔM/ΔO/Δg/e` 可稳定读出。  
2. `M_obj` 可用。  
3. operator / binding / signature 初版可用。  
4. `Δρ` 不是纯噪声。  

否则 event proposal 会把：
- 表征噪声  
- 对象化失败  
- 普通 dynamics 残差

混当成“事件”，导致 buffer 污染。

---

## 9. 多游戏训练与泛化验证计划（围绕 Phase 1A / 1B / 2）

## 9.1 训练顺序建议

### Step A：micro-suite / toy tasks
先做少量验证任务，专门测：
- 局部移动  
- block / wall / passability  
- key-door / unlock  
- push / pickup  
- 局部区域 hazard / reward  
- 简单 topology open / close

这些任务不是最终 benchmark，而是结构接口与规则层的验证基座。

### Step B：单游戏长跑
在 Atari 和 ARC3 各挑少量代表游戏，分别验证：
- world model 稳定性  
- slot/objectification 可用性  
- operator/binding/signature 可用性

### Step C：多游戏混训
开始做：
- within-domain 多游戏训练  
- Atari / ARC3 domain-balanced 混训  
- unseen-game warm-start 验证

## 9.2 泛化验证维度

### A. 单游戏内泛化
- 同游戏不同种子 / 不同初始条件  
- 同游戏不同 level / phase

### B. 同域跨游戏泛化
- Atari 内不同游戏  
- ARC3 内不同游戏

### C. 跨域迁移
- 从 Atari 学到的 effect / operator 先验是否改善 ARC3 中类似结构  
- 从 ARC3 学到的 map/object 组织是否改善 Atari 小局部结构理解

### D. 冷启动 vs warm-start
同一新游戏上比较：
- 从零训练  
- 带 prior bank 初始化  
- 带已有 operator 初始化

---

## 10. 代码目录建议

在当前 `r2dreamer` 结构上，建议新增：

```text
.
├── train.py
├── trainer.py
├── dreamer.py
├── rssm.py
├── buffer.py
├── networks.py
├── structured_readout.py        # x_t = (M,O,g,rho)
├── effect_model.py              # z_eff
├── effect_heads.py              # ΔM, ΔO, Δg, e
├── objectification.py           # Phase 1B 对象化过程训练
├── operator_bank.py             # Phase 2 operator bank
├── binding_head.py              # Phase 2 binding
├── signature_head.py            # Phase 2 signature
├── reachability_head.py         # weak / strong reachability
├── goal_progress_head.py        # weak / strong goal progress
├── utils/
│   ├── slot_matching.py
│   ├── replay_sampling.py
│   ├── validation_metrics.py
│   └── phase_gates.py
├── tests/
│   ├── test_shapes.py
│   ├── test_effect_model.py
│   ├── test_objectification.py
│   ├── test_operator_bank.py
│   ├── test_phase_gates.py
│   └── smoke/
├── configs/
│   ├── exp/
│   ├── env/
│   ├── model/
│   └── trainer/
└── scripts/
    ├── train_phase1a_*.sh
    ├── train_phase1b_*.sh
    ├── train_phase2_*.sh
    └── eval_phase*.sh
```

说明：
- 尽量保持对现有骨架的最小侵入。  
- 新模块尽量独立文件，方便 Codex 分阶段实现。  

---

## 11. 交给 Codex 的实现方式建议

## 11.1 一次只交付一个 phase
不要让 Codex 一次改完全部系统。  
建议按以下顺序分批交付：

1. Phase 1A skeleton + tests  
2. Phase 1A training/eval  
3. Phase 1B objectification  
4. Phase 2 operator/binding/signature  
5. Phase 3 event pipeline

## 11.2 每次交付都必须包含

### A. 代码
- 模块实现  
- 配置项  
- 训练脚本入口

### B. 测试
- 单元测试  
- smoke test  
- 形状和 checkpoint 测试

### C. 运行结果
- 至少一个短跑日志  
- 关键 loss 曲线  
- 验收指标表

### D. 结论
- 是否通过当前 phase 的 gate  
- 若未通过，卡在哪个指标

## 11.3 交付模板建议

Codex 每轮交付最好固定成：

1. 改动文件列表  
2. 新增配置项  
3. 新增脚本命令  
4. 已跑测试  
5. 已跑训练命令  
6. 指标结果  
7. 是否达到 phase gate  

---

## 12. Phase Gate 总表

| Phase | 核心目标 | 必须通过的 gate |
|---|---|---|
| 1A | 主干 + 双视图 + effect baseline | 主干稳定；`M/O/g/rho_weak` 可读；`ΔM/ΔO/Δg/e` 优于简单 baseline |
| 1B | 对象化过程 | `M_obj` 可用；slot 匹配与局部作用性明显优于随机；主干不退化 |
| 2 | operator / binding / signature 初版 | operator 非塌缩；binding/signature 非纯噪声；`Δρ` 可用；可作为 event pipeline 输入 |
| 3 | 事件时间尺度拆分 | event proposal 质量可用；event / long-horizon buffer 标签稳定 |
| 4 | 跨游戏先验沉淀 | warm-start 优于冷启动；prototype merge/split 可控 |
| 5 | Selective SL | 只在高价值节点触发；信息收益为正；训练预算可控 |

---

## 13. 公式与词汇表

### 13.1 公式顺序说明

#### F1
$$s_t=(h_t,z_t)$$
Dreamer 核心 latent。

#### F2
$$x_t=(M_t,O_t,g_t,\rho_t)$$
结构读出状态。

#### F3
$$z_t^{eff}=E(s_t,a_t,\rho_t)$$
统一 effect latent。

#### F4
$$c_t=(\Delta M_t,\Delta O_t,\Delta g_t,\Delta \rho_t,e_t,\omega_t,d_t,\kappa_t)$$
动作后果解释接口。

#### F5
$$r_t=(u_t,b_t,\sigma_t)$$
规则原型表示。

#### F6
$$P_{evt}(t)=\alpha U_{explicit}+\beta U_{residual}+\gamma((1-M_{obj})U_{struct\_proxy}+M_{obj}U_{struct})+\delta U_{rule}$$
事件 proposal 分数。

#### F7
$$S_{SL}=M_{obj}\cdot A_{task}\cdot C_{diag}\cdot G_{future}$$
Selective SL 触发分数。

### 13.2 词汇表

| 词汇 / 符号 | 含义 |
|---|---|
| `s_t` | Dreamer 核心 latent state |
| `h_t` | recurrent / deterministic memory |
| `z_t` | stochastic latent |
| `x_t` | 结构读出状态 |
| `M_t` | map/grid/region view |
| `O_t` | object/slot view |
| `g_t` | global state |
| `ρ_t` | 规则上下文或规则后验压缩表示 |
| `z_eff` | effect latent |
| `c_t` | effect packet |
| `ΔM_t` | 地图/区域结构变化 |
| `ΔO_t` | 对象/slot 变化 |
| `Δg_t` | 全局状态变化 |
| `Δρ_t` | 规则后验变化 |
| `e_t` | 事件摘要 token |
| `ω_t` | 作用范围 / binding 权重 |
| `d_t` | 持续时间 |
| `κ_t` | 长期影响 / future divergence |
| `u_t` | operator prototype |
| `b_t` | binding scope |
| `σ_t` | spatiotemporal signature |
| `M_obj` | 对象化成熟度 |
| `P_evt` | 事件候选分数 |
| `S_SL` | Selective SL 触发分数 |

---

## 14. 最后一页的执行建议

如果马上开始实现，我建议严格按下面顺序交给 Codex：

### 第一步
先做 **Phase 1A skeleton**：
- `structured_readout.py`  
- `effect_model.py`  
- `effect_heads.py`  
- `reachability_head.py`  
- `goal_progress_head.py`  
- 对应配置与 shape tests

### 第二步
补完 **Phase 1A 训练与评估**：
- 训练脚本  
- smoke run  
- Phase 1A gate 验收

### 第三步
做 **Phase 1B 对象化过程训练**：
- `objectification.py`  
- `slot_matching.py`  
- `M_obj` 产出  
- Phase 1B gate

### 第四步
做 **Phase 2 operator/binding/signature**：
- `operator_bank.py`  
- `binding_head.py`  
- `signature_head.py`  
- `Δρ` 读出  
- Phase 2 gate

直到这些通过，再谈 event pipeline 与 SL。

这就是当前最稳、最符合前面对话共识的实现路线。
