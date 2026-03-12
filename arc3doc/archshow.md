## X. Phase2 结构总览图（最新版）

本节给出当前 Phase2 的结构总览。
这些图覆盖了当前实现中的：

1. 从 Dreamer 主干到 Phase2 的输入来源；
2. Phase2 的 heads / retrieval / apply 前向主链；
3. RuleMemory 的写入、freshness 与 support 结构；
4. 2-step / 4-step / 7-step rollout、curriculum、gate 与 signoff 的关系。

需要特别说明的是：

> 当前 Phase2 已经不是“只有监督头”的状态，
> 而是已经具备：
> - RuleMemory
> - retrieval / apply
> - one-step `rho` apply
> - 2/4/7-step `rho` shadow rollout
> - curriculum 与 rollout gate
>
> 但当前 rollout 仍然是 **`rho-only shadow rollout`**，
> 不是 full structured rollout，也不是 full latent rollout。

---

### 图 X-1：从 Dreamer 主干到 Phase2 的输入来源

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Dreamer / RSSM 主干（world model）                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：                                                                       │
│ - image / observation                                                        │
│ - previous action                                                            │
│ - previous latent state                                                      │
│                                                                              │
│ 输出：                                                                       │
│ - feat_seq                # 当前时序 latent feature                          │
│ - stoch / deter           # RSSM 状态                                        │
│ - reward / cont / actor / value 路径                                         │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Phase 1A: StructuredReadout                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：                                                                       │
│ - feat_seq[:, :-1]          # 当前步特征                                     │
│ - feat_seq[:, 1:]           # 下一步特征                                     │
│                                                                              │
│ 输出（current / nxt 两套）                                                   │
│ - M_t      # map / region view                                              │
│ - O_t      # object / slot view                                             │
│ - g_t      # global view                                                    │
│ - rho_t    # rule context                                                   │
│                                                                              │
│ 还会构造：                                                                   │
│ - target_delta_M   = nxt.M_t  - current.M_t                                  │
│ - target_delta_O   = nxt.O_t  - current.O_t                                  │
│ - target_delta_g   = nxt.g_t  - current.g_t                                  │
│ - target_delta_rho = nxt.rho_t - current.rho_t                               │
│ - transition masks / valid ratio                                             │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Phase 1A: EffectModel / z_eff                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：                                                                       │
│ - feat_seq[:, :-1]                                                            │
│ - data["action"]                                                              │
│ - current["rho_t"]                                                            │
│                                                                              │
│ 输出：                                                                       │
│ - z_eff    # 统一动作后果 latent                                             │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Phase 1B: Objectification                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：                                                                       │
│ - current["O_t"]                                                              │
│ - nxt["O_t"]                                                                  │
│ - z_eff                                                                       │
│ - target_delta_O                                                              │
│ - event_target / transition masks                                             │
│                                                                              │
│ 输出：                                                                       │
│ - objectification_out                                                         │
│   至少包括：                                                                  │
│   - objectness_score                                                          │
│   - slot_match / slot_match_margin                                            │
│   - slot_cycle / slot_identity / object_interface 等                          │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Phase 2 输入准备层                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ Phase2 直接吃的输入：                                                         │
│ - feat                  = structured["feat_seq"][:, :-1]                     │
│ - action                = data["action"]                                     │
│ - current["M_t"] / current["O_t"] / current["g_t"] / current["rho_t"]        │
│ - z_eff                                                                     │
│ - objectification_out                                                        │
│                                                                              │
│ Phase2 监督 target 还会从这里来：                                             │
│ - binding_target     = _binding_proxy(structured)                            │
│ - scope/duration/impact target = _signature_proxy(structured)                │
│ - target_delta_rho   = structured["target_delta_rho"]                        │
│ - next rho target    = structured["nxt"]["rho_t"]                            │
└──────────────────────────────────────────────────────────────────────────────┘



图注 X-1
该图说明了 Phase2 不是直接从像素进入，而是建立在 Dreamer latent、结构读出 M_t / O_t / g_t / rho_t、统一动作后果 z_eff、以及对象化输出之上的高层接口层。

图 X-2：Phase2 前向执行主链（heads / retrieval / apply）



┌──────────────────────────────────────────────────────────────────────────────┐
│                      Phase2 输入：feat / action / current / z_eff            │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ OperatorBank                                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：feat, action, M_t, O_t, g_t, rho_t, z_eff                              │
│ 输出：                                                                       │
│ - q_u_logits                                                                  │
│ - q_u               # operator posterior                                     │
│ - target_q          # target operator distribution                           │
│ - operator_embed                                                               │
│ - context_embed                                                                │
│ - effect_embed                                                                 │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
               ▼                                ▼
┌────────────────────────────┐      ┌──────────────────────────────┐
│ BindingHead                │      │ SignatureHead                 │
├────────────────────────────┤      ├──────────────────────────────┤
│ 输入：operator_embed,      │      │ 输入：operator_embed,        │
│      context_embed         │      │      context_embed           │
│ 输出：q_b                  │      │ 输出：q_sigma, scope,        │
│                            │      │      duration, impact        │
└──────────────┬─────────────┘      └──────────────┬───────────────┘
               │                                   │
               └───────────────┬───────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RuleUpdateHead                                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：z_eff, operator_embed, q_b, q_sigma                                    │
│ 输出：delta_rule_pred                                                        │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RuleMemory.retrieve(q_u, q_b, q_sigma)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输出：                                                                       │
│ - memory_delta_rule                                                          │
│ - memory_signature_proto                                                     │
│ - memory_conf                                                                │
│ - memory_weights                                                             │
│ - memory_top_weight                                                          │
│ - memory_prior_scale                                                         │
│ - memory_retrieve_temperature                                                │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Gate：_phase2_gate(objectification_out)                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输出：                                                                       │
│ - gate                                                                        │
│ - object_gate                                                                 │
│ - match_gate                                                                  │
│ - warmup_gate                                                                 │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RuleApply                                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ 输入：                                                                       │
│ - current["rho_t"]                                                            │
│ - delta_rule_pred                                                             │
│ - memory_delta_rule                                                           │
│ - operator_conf                                                               │
│ - binding_conf                                                                │
│ - memory_conf                                                                 │
│ - gate                                                                        │
│                                                                              │
│ 输出：                                                                       │
│ - delta_rule_fused                                                            │
│ - rho_next_pred                                                               │
│ - fusion_alpha                                                                │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Phase2Artifact                                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ 当前 artifact 打包了：                                                       │
│ - q_u / q_b / q_sigma                                                        │
│ - operator_id / binding_id / conf                                            │
│ - delta_rule_pred                                                            │
│ - memory_delta_rule / memory_conf                                            │
│ - delta_rule_fused                                                           │
│ - rho_next_pred                                                              │
│ - gate / object_gate / match_gate / warmup_gate                              │
└──────────────────────────────────────────────────────────────────────────────┘


图注 X-2
该图展示当前 Phase2 的前向执行主链。Phase2 已经不再是纯监督头，而是形成了：
heads -> RuleMemory.retrieve -> RuleApply -> rho_next_pred 的可执行规则接口。

图 X-3：RuleMemory 写入、freshness 与 support 结构


┌──────────────────────────────────────────────────────────────────────────────┐
│                   RuleMemory：operator × binding 网格                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ 每个 cell 存：                                                               │
│ - delta_rule_proto                                                           │
│ - signature_proto                                                            │
│ - ema_conf                                                                   │
│ - write_mass                                                                 │
│ - support_ema                                                                │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RuleMemory.retrieve(...)                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│ 不只看 q_u × q_b，还看：                                                     │
│ - support_prior   # 来自 support_ema                                         │
│ - conf_prior      # 来自 ema_conf                                            │
│ - signature_score # 来自 q_sigma 与 signature_proto 的相似度                 │
│ - effective_temperature                                                      │
│                                                                              │
│ 这就是 retrieval calibration：                                               │
│ 在读取时做“近期活跃度 + 历史可靠度 + 当前匹配度”的联合校正。                 │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RuleMemory.update(...)                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│ 写入需要 write_mask：                                                        │
│ - operator_conf 足够                                                         │
│ - binding_conf 足够                                                          │
│ - delta_alignment 足够                                                       │
│ - apply_error 足够小                                                         │
│ - gate / valid mask 通过                                                     │
│                                                                              │
│ 更新内容：                                                                   │
│ - prototype_decay / min_blend                                                │
│ - ema_conf                                                                   │
│ - support_ema                                                                │
│ - write_mass / usage_count                                                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Fresh usage / support surface                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ valid_cells = support_ema > support_min                                      │
│                                                                              │
│ fresh_usage_fraction                                                         │
│   = 当前 memory 网格里“仍然活着的 cell 占比”                                │
│                                                                              │
│ 注意：                                                                       │
│ 这里的“fresh”不是环境还有没有新奇事件，                                      │
│ 而是 memory 里当前仍然被持续支持、持续可用的活跃支持面。                     │
└──────────────────────────────────────────────────────────────────────────────┘


图注 X-3
该图说明 RuleMemory 不只是简单缓存，而是一个带有 retrieval calibration、prototype 更新、freshness 追踪与活跃支持面的规则记忆模块。
当前 2-step / 4-step / 7-step rollout 的稳定性，与 RuleMemory 的支持面是否仍然活跃高度相关。

图 X-4：2-step / 4-step / 7-step、curriculum、gate 与 signoff



┌──────────────────────────────────────────────────────────────────────────────┐
│                       One-step rule apply                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ rho_t                                                                        │
│   -> heads -> retrieve -> apply -> rho_next_pred                            │
│                                                                              │
│ loss:                                                                       │
│ - rule_update                                                                │
│ - memory_read_loss                                                           │
│ - memory_agreement_loss                                                      │
│ - rule_apply_loss                                                            │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           2-step rho rollout                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ 用 predicted rho 继续滚下一步：                                              │
│ - two_step_memory_conf                                                       │
│ - two_step_retrieval_agreement                                               │
│ - two_step_apply_error                                                       │
│                                                                              │
│ 这是判断“rule path 是否足够稳”的第一层。                                    │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Four-step curriculum gate                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ 读取这些指标：                                                               │
│ - two_step_memory_conf                                                       │
│ - two_step_retrieval_agreement                                               │
│ - two_step_apply_error                                                       │
│ - rule_memory_usage                                                          │
│ - rule_memory_fresh_usage                                                    │
│ - rule_apply_error                                                           │
│ - four_step_apply_error                                                      │
│ - seven_step_apply_error                                                     │
│                                                                              │
│ enable 条件：                                                                │
│ 2-step 稳 + memory 活跃 + fresh usage 够 + apply 误差够小                    │
│                                                                              │
│ disable 条件：                                                               │
│ retrieval 掉且 fresh usage 掉                                                │
│ 或 four/seven-step error 过大                                                │
│                                                                              │
│ 输出：                                                                       │
│ - four_step_curriculum_active                                                │
│ - four_step_curriculum_scale                                                 │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         4-step / 7-step rollout                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ 4-step：                                                                     │
│ - 受 curriculum 调制                                                         │
│ - four_step_apply 可进入训练                                                 │
│                                                                              │
│ 7-step：                                                                     │
│ - 主要用于更长 horizon 的 stress test                                        │
│ - 也为 disable 条件提供证据                                                  │
│                                                                              │
│ 当前这些 rollout 都属于：                                                    │
│ - rho-only shadow rollout                                                    │
│ 不是 full structured rollout，也不是 full latent rollout。                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Signoff 层                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ evaluate_phase2_executable_gate()                                            │
│   -> one-step retrieval/apply 是否 ready                                     │
│                                                                              │
│ evaluate_phase2_rollout_two_step_gate()                                      │
│   -> 2-step 是否 ready                                                       │
│                                                                              │
│ evaluate_phase2_rollout_long_gate()                                          │
│   -> 4-step / 7-step 是否进入可接受区                                        │
│                                                                              │
│ evaluate_phase2_rollout_gate()                                               │
│   -> rollout ready 组合                                                     │
│                                                                              │
│ evaluate_atari_closed_loop()                                                 │
│   -> executable + rollout + task + baseline_relative                         │
└──────────────────────────────────────────────────────────────────────────────┘



图注 X-4
当前 Phase2 的多步 rollout 不是单独存在的，而是和 retrieval calibration、memory freshness、curriculum、gate、signoff 深度绑定的一套训练与验收系统。
其核心逻辑是：
先保证 one-step 与 2-step 稳，再逐步开放 4-step；7-step 主要用于更长 horizon 的压力测试与退火/关闭证据。