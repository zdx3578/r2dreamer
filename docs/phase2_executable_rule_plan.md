# Phase2 Executable Rule Interface Plan

## Goal

Make Phase2 executable at one step instead of keeping it as a collection of supervision heads.

Closed loop:

`current structured state -> Phase2 artifact -> RuleMemory retrieve -> rule fusion -> rho_next_pred -> align with next_rho`

The first implementation is limited to the `rho` channel. It does not modify RSSM state transitions, actor, or planner.

## Accepted Decisions

1. `RuleMemory` is indexed by `operator x binding`, not by `operator` alone.
2. `signature` is stored as a memory value and used as a retrieval aid, but it is not part of the primary key in v1.
3. Phase2 is split into:
   - `_phase2_forward(structured, data) -> Phase2Artifact`
   - `_phase2_losses(artifact, structured) -> losses, metrics`
   - `_phase2_memory_update(artifact, structured) -> metrics`
4. The main new executable supervision is `rule_apply_loss`, which aligns `rho_next_pred` with `next_rho`.
5. Memory writes are no-grad EMA updates and are only allowed when:
   - phase2 gate is open
   - event target is active
   - operator confidence is above threshold
   - binding confidence is above threshold

## Adjustments For This Repo

1. Loss scales stay in `loss_scales`, so `rule_apply` is added there instead of adding a separate `phase2.apply_loss_scale`.
2. Retrieval uses soft `operator x binding` weights over populated memory cells. Writes use top-1 cells only.
3. The first retrieval implementation uses `signature` to modulate retrieval confidence instead of making it a hard lookup dimension.
4. `phase2.match_margin_threshold` is already deprecated in behavior because match gating is disabled. It stays untouched in this step and should be removed or explicitly deprecated later.

## Phase2Artifact v1

Phase2Artifact carries the explicit Phase2 interface:

- `q_u`, `operator_id`, `operator_conf`, `operator_embed`
- `q_b`, `binding_id`, `binding_conf`
- `q_sigma`, `scope`, `duration`, `impact`
- `delta_rule_pred`
- `memory_delta_rule`, `memory_signature_proto`, `memory_conf`
- `delta_rule_fused`, `rho_next_pred`
- gate scalars for logging

It also keeps auxiliary tensors required by losses, such as `target_q`, `context_embed`, and `effect_embed`.

## New Modules

### `rule_memory.py`

Responsibilities:

- store EMA prototypes for `delta_rule` and `signature`
- maintain usage and confidence statistics
- retrieve memory values from `operator x binding`
- update memory values with gated EMA writes

Stored state per cell:

- `delta_rule_proto`
- `signature_proto`
- `usage_count`
- `ema_conf`

### `rule_apply.py`

Responsibilities:

- fuse direct `delta_rule_pred` with `memory_delta_rule`
- predict `rho_next_pred = rho_t + delta_rule_fused`

The first fusion rule is a deterministic confidence gate, not a learned network.

## Metrics To Add

- `phase2/operator_top1_conf`
- `phase2/binding_top1_conf`
- `phase2/memory_conf`
- `phase2/retrieval_agreement`
- `phase2/rule_apply_error`
- `phase2/rule_memory_usage`
- `phase2/rule_memory_entropy`
- `phase2/rule_memory_write_rate`
- `phase2/pred_delta_rule_abs`
- `phase2/retrieved_delta_rule_abs`
- `phase2/fused_delta_rule_abs`

## Implementation Order

1. Add `Phase2Artifact`, `RuleMemory`, and `RuleApply`.
2. Refactor `dreamer.py` to build and use `Phase2Artifact`.
3. Add `rule_apply_loss` and memory update path.
4. Add unit tests for memory retrieval, memory update gating, and rule fusion.
5. Extend runtime metrics and monitoring.
