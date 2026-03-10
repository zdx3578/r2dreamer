# Phase2 Next Stage Plan

## Current Baseline

Phase2 has already moved from "supervision heads only" to a one-step executable rule path inside training:

`structured state -> Phase2 artifact -> RuleMemory retrieve -> RuleApply fuse -> rho_next_pred -> align with next_rho`

This path is now stable on the current baseline run:

- Run: `logdir/verify_exec_v5_30k_b4_alien_2bc36b5`
- `phase2_executable.ready = true`
- `atari_closed_loop.ready = true`
- `retrieval_agreement_mean = 0.9987801313400269`
- `operator_top1_conf_mean = 0.4073645234107971`
- `binding_top1_conf_mean = 0.7520877480506897`
- `memory_conf_mean = 0.9569507241249084`
- `rule_apply_error_mean = 0.0004358673933893442`
- `slot_match_mean = 0.5093164443969727`
- `slot_identity_mean = 0.5624032020568848`
- `object_interface_mean = 0.6171978712081909`
- `ret_mean = 1.0185977280139924`
- `score_mean = 209.0909090909091`
- `score_max = 360.0`

What is done:

- explicit Phase2 artifact exists
- `operator x binding` RuleMemory exists
- one-step rule retrieval and fusion exists
- one-step rule application is trained and monitored
- executable gate and task gate both pass on the current 30k baseline

What is still missing:

- Phase2 outputs are still mostly consumed by auxiliary losses
- RuleMemory prototype updates are not truly forgetful yet
- memory write decisions still depend mostly on confidence, not write quality
- `gate == 0` does not yet fully neutralize memory influence in `RuleApply`
- there is no controlled multi-step rollout consumer yet
- executable metrics and task metrics are not yet summarized together as a regression guard in monitoring

## Long-Term Roadmap

### Phase A: Stable Executable Core

Goal: keep one-step rule execution correct, writable, and correctable.

This phase focuses on:

- RuleMemory forgetting and correction
- write quality gating
- safe `RuleApply` gate semantics
- non-regression against the current 30k baseline

### Phase B: Controlled Inference Consumer

Goal: let the rule interface affect a limited inference path without touching RSSM state transitions, actor, or planner.

This phase focuses on:

- shadow rule rollout on the `rho` channel
- two-step rule execution metrics
- optional effect-model side conditioning using `delta_rule_fused` or `rho_next_pred`

### Phase C: Task Closed Loop

Goal: verify that executable rules help or at least do not harm task learning.

This phase focuses on:

- tying executable gates to `atari_task` and `atari_closed_loop`
- adding baseline-relative summaries to monitoring
- checking that rule improvements do not hide task regressions

### Phase D: Multi-Step Symbolic Rollout

Goal: move from one-step execution to short-horizon symbolic rollout.

This phase is only valid after Phases A-C are stable.

### Phase E: Planner Or Policy Integration

Goal: allow downstream planning or control to consume rule outputs.

This phase stays last. It should not start before multi-step rollout is stable.

## Accepted And Deferred Decisions

Accepted now:

- keep the primary RuleMemory index as `operator x binding`
- keep `signature` as a retrieval aid and stored value, not as a primary key
- keep the next consumer limited to the `rho` channel
- keep planner and actor/value integration out of scope for the next stage
- treat `slot_concentration` as a secondary workstream, not the main blocker for the next Phase2 step

Adjusted from v1:

- RuleMemory prototypes need explicit forgetting instead of pure cumulative blending
- memory writes need quality gating, not only confidence gating
- `phase2_executable` alone is not enough for sign-off; monitoring must also track `atari_closed_loop`

Deferred:

- `operator x binding x signature` multi-prototype memory
- direct RSSM latent overwrites
- direct actor/value conditioning
- planner integration

## Integrated Next 5-Step Plan

### Step 1: Stabilize RuleMemory Prototypes

Goal:

- make memory correctable after early bad writes
- make the prototype update rule match the intended "EMA-like but revisable" behavior

Changes:

- split prototype update behavior from support statistics
- add an explicit prototype forgetting parameter such as `prototype_decay` or `prototype_momentum`
- keep `write_mass`, `usage_count`, and `ema_conf` as support statistics
- keep retrieval valid-cell logic based on written support, but stop treating support accumulation as the prototype update rule itself

Expected files:

- `rule_memory.py`
- `configs/model/_base_.yaml`
- `tests/test_phase2_rule_execution.py`

Required new tests:

- bad early prototype can be corrected by later writes
- the same cell can move toward a new rule target after repeated better writes

### Step 2: Tighten Memory Write Quality Gates

Goal:

- reduce dirty writes early in training

Changes:

- keep current event and confidence gates
- add at least one quality condition before final write, chosen from:
  - `rule_apply_error < threshold`
  - cosine agreement between `delta_rule_pred` and `target_delta_rho` above threshold
  - `memory_read_error < threshold`
- keep writes no-grad and top-1 per sample

Expected files:

- `dreamer.py`
- `configs/model/_base_.yaml`
- `tests/test_phase2_rule_execution.py`

Required new tests:

- low-quality samples do not write
- high-confidence but wrong-direction samples do not write

### Step 3: Fix RuleApply Gate Semantics

Goal:

- ensure `gate == 0` means no rule execution effect

Changes:

- when `gate == 0`, prevent memory from affecting the fused delta
- preferred safe behavior:
  - `delta_rule_fused = 0`
  - `rho_next_pred = rho_t`
- alternative acceptable behavior:
  - `delta_rule_fused = delta_rule_pred`
  - memory path is disabled

Expected files:

- `rule_apply.py`
- `tests/test_phase2_rule_execution.py`

Required new tests:

- gate-off apply does not consume memory
- gate-off apply leaves `rho_next_pred` unchanged if the chosen design is zero-effect

### Step 4: Add Two-Step Rho Rollout As The First Controlled Consumer

Goal:

- let Phase2 outputs participate in a limited inference path before touching RSSM, actor, or planner

Changes:

- add a two-step shadow rollout on the `rho` channel
- make `two-step rho rollout` the first concrete implementation of the shadow rollout path
- add `two_step_apply_error` as the main new executable rollout loss
- add rollout metrics such as:
  - `phase2/two_step_apply_error`
  - `phase2/two_step_delta_rule_abs`
  - `phase2/two_step_memory_conf`
- optionally add a small effect-side consumer that sees `delta_rule_fused` or `rho_next_pred` as an auxiliary condition
- add a new gate such as `phase2_rollout_ready`

Expected files:

- `dreamer.py`
- `phase2_artifact.py`
- `utils/phase_gates.py`
- `scripts/monitor_seed_runs.py`
- `tests/test_phase2_rule_execution.py`
- `tests/test_phase_gates.py`

Required new tests:

- two-step rollout error is finite and decreases on controlled fixtures
- gate-off path still does not leak memory into rollout

### Step 5: Tie Executable Rules To Task Closed Loop

Goal:

- prevent internal Phase2 progress from hiding task regressions

Changes:

- extend monitoring to report:
  - `phase2_executable`
  - `phase2_rollout_ready`
  - `atari_task`
  - `atari_closed_loop`
  - baseline-relative summaries versus the current 30k baseline run
- mark a run healthy only if both rule execution and task signals remain healthy

Expected files:

- `scripts/monitor_seed_runs.py`
- `utils/phase_gates.py`
- `tests/test_monitor_seed_runs.py`
- `tests/test_phase_gates.py`

Required new tests:

- executable-ready but task-bad should fail the combined summary
- task-good but rollout-bad should fail the combined summary

## Validation Ladder

The next stage should be validated in this order:

### 1. Unit And Integration Tests

Must pass before training:

- `tests/test_phase2_rule_execution.py`
- `tests/test_phase_gates.py`
- `tests/test_monitor_seed_runs.py`

### 2. Short 7k Filter

Purpose:

- reject obviously bad memory or gate changes quickly

Minimum acceptance:

- `phase2_executable.ready = true`
- `retrieval_agreement_mean >= 0.75`
- `rule_apply_error_mean <= 0.05`

### 3. Mid 16k Stability Check

Purpose:

- verify that the mechanism stays stable after the early warmup phase

Minimum acceptance:

- `phase2_executable.ready = true`
- `retrieval_agreement_mean >= 0.90`
- `rule_apply_error_mean <= 0.01`
- if Step 4 is implemented, `phase2_rollout_ready = true`

### 4. Full 30k Benchmark

Purpose:

- verify no structural regression against the current reference run

Minimum acceptance:

- `atari_closed_loop.ready = true`
- `ret_mean >= 0.80`
- `score_mean >= 150.0`
- `slot_match_mean >= 0.45`
- `rule_apply_error_mean <= 0.01`

Reference baseline to compare against:

- `retrieval_agreement_mean = 0.9987801313400269`
- `rule_apply_error_mean = 0.0004358673933893442`
- `ret_mean = 1.0185977280139924`
- `score_mean = 209.0909090909091`
- `slot_match_mean = 0.5093164443969727`

## Immediate Execution Order

The next implementation turn should follow this exact order:

1. Step 1: RuleMemory prototype stabilization
2. Step 2: write quality gating
3. Step 3: RuleApply gate fix
4. Step 4: two-step rho rollout, shadow rollout metrics, and rollout gate
5. Step 5: combined executable plus task monitoring

Planner, actor/value integration, and multi-step symbolic control stay out of scope until all five items above are complete.
