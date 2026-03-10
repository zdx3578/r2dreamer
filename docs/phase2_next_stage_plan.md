# Phase2 Next Stage Plan

## Current Status

Phase2 is no longer at the "supervision heads only" stage.
The current codebase already has a writable and executable rule path:

`structured state -> Phase2 artifact -> RuleMemory retrieve -> RuleApply fuse -> rho_next_pred`

Two reference runs anchor the current stage:

- executable reference: `logdir/verify_exec_v5_30k_b4_alien_2bc36b5`
- rollout reference: `logdir/verify_rollout_v2_30k_b4_alien_304f8ba`

The executable reference established that one-step rule execution is stable on Atari:

- `phase2_executable.ready = true`
- `atari_closed_loop.ready = true`
- `retrieval_agreement_mean = 0.9987801313400269`
- `rule_apply_error_mean = 0.0004358673933893442`
- `ret_mean = 1.0185977280139924`
- `score_mean = 209.0909090909091`

The rollout reference established that longer shadow rollout metrics are already stable enough to move beyond pure monitoring:

- `phase2_rollout.ready = true`
- `two_step_apply_error_mean ~= 4.13e-4`
- `four_step_apply_error_mean ~= 6.54e-4`
- `seven_step_apply_error_mean ~= 1.25e-3`
- `ret_mean ~= 1.02`
- `score_mean = 216.0`

## Old Step 1-5 Status

The previous five-step plan is mostly implemented and should no longer be read as future work.

### Step 1: RuleMemory Prototype Stabilization

Status: delivered.

What landed:

- prototype updates are no longer pure cumulative overwrite
- repeated good writes can correct an early bad prototype
- lifetime support statistics remain tracked separately from the prototype value

### Step 2: Memory Write Quality Gating

Status: delivered.

What landed:

- writes already require confidence
- writes already require quality filters such as apply error and delta alignment
- monitor metrics already report write alignment, write apply error, and write quality rate

### Step 3: RuleApply Gate Semantics

Status: delivered.

What landed:

- `gate == 0` zeros the fused rule delta
- `rho_next_pred = rho_t` when the gate is off
- memory does not leak into the apply path when the gate is closed

### Step 4: Controlled Rho Rollout Consumer

Status: delivered and expanded.

What landed:

- shadow rollout now records `2/4/7-step` metrics on the `rho` channel
- `two_step_apply` is trained
- `four_step_apply` is now part of training with a small loss weight
- `seven_step` remains monitor-only

What is still true:

- this is still a `rho-only` shadow rollout
- `M_t / O_t / g_t / feat / action` remain teacher-forced
- this is not yet a full structured or full latent rollout

### Step 5: Closed-Loop Monitoring

Status: delivered in basic form and now being tightened.

What landed:

- seed summaries already report `phase2_executable`, `phase2_rollout`, `atari_task`, and `atari_closed_loop`
- aggregate summaries already collect readiness flags and peak metrics

What was missing and is now the active upgrade path:

- rollout readiness was still effectively `two-step ready`
- `atari_closed_loop` previously only checked `phase2_ready + task_ready`
- monitoring leaned too heavily on peak metrics for rollout sign-off

## Current Four-Step Plan

The old Step 1-5 sequence should now be treated as history.
The current Phase2 workstream is the narrower four-step plan below.

### 1. RuleMemory Freshness

Goal:

- make retrieval prefer recently supported cells instead of permanently favoring old high-write cells

Implementation direction:

- keep `usage_count` and `write_mass` as lifetime monitoring statistics
- add freshness-aware support such as `support_ema`
- use freshness-aware support for retrieval validity and retrieval prior
- keep prototype correction separate from retrieval support

Expected files:

- `rule_memory.py`
- `dreamer.py`
- `configs/model/_base_.yaml`
- `tests/test_phase2_rule_execution.py`

### 2. Rollout Gate Upgrade

Goal:

- stop treating rollout readiness as "two-step only"

Implementation direction:

- split rollout readiness into:
  - `phase2_rollout_two_step_ready`
  - `phase2_rollout_long_ready`
- keep `phase2_rollout.ready` as the combined sign-off
- require `four_step_*` to be healthy for long-horizon readiness
- require `seven_step_apply_error` to stay bounded instead of only logging it

Expected files:

- `utils/phase_gates.py`
- `tests/test_phase_gates.py`

### 3. Four-Step Training

Goal:

- move long-horizon rollout from monitor-only toward a light curriculum

Implementation direction:

- keep `two_step_apply` as the dominant rollout training loss
- add `four_step_apply` with a small loss weight
- keep `seven_step` monitor-only for now

Expected files:

- `dreamer.py`
- `configs/model/_base_.yaml`
- `tests/test_phase1a.py`

### 4. Closed-Loop And Monitor Upgrade

Goal:

- make rollout and executable quality part of Atari sign-off
- make end-window behavior more important than isolated peaks

Implementation direction:

- upgrade `atari_closed_loop` to require:
  - `phase2_executable_ready`
  - `phase2_rollout_ready`
  - `task_ready`
- keep peak summaries, but stop treating them as the main rollout decision signal
- add end-window summaries and baseline-relative deltas to monitoring

Expected files:

- `utils/phase_gates.py`
- `scripts/monitor_seed_runs.py`
- `tests/test_phase_gates.py`
- `tests/test_monitor_seed_runs.py`

## Scope Boundaries

Still in scope:

- Atari-only Phase2 stabilization
- executable rule retrieval
- freshness-aware memory support
- short-horizon `rho` rollout
- monitoring and closed-loop regression guards

Still out of scope:

- planner integration
- direct actor/value conditioning
- direct RSSM latent overwrite
- ARC3 re-expansion
- full structured or full latent rollout

## Validation Ladder

### 1. Unit And Integration Tests

Must pass before training:

- `tests/test_phase2_rule_execution.py`
- `tests/test_phase_gates.py`
- `tests/test_monitor_seed_runs.py`

### 2. Short 7k Filter

Purpose:

- catch obviously bad freshness or rollout changes early

Minimum acceptance:

- `phase2_executable.ready = true`
- `phase2_rollout_two_step_ready = true`
- `retrieval_agreement_mean >= 0.75`
- `rule_apply_error_mean <= 0.05`

### 3. Mid 16k Stability Check

Purpose:

- verify that freshness and rollout stay stable after warmup

Minimum acceptance:

- `phase2_executable.ready = true`
- `phase2_rollout_two_step_ready = true`
- `phase2_rollout_long_ready = true`
- `retrieval_agreement_mean >= 0.90`
- `rule_apply_error_mean <= 0.01`

### 4. Full 30k Benchmark

Purpose:

- verify no regression against the current Atari-only Phase2 baseline

Minimum acceptance:

- `atari_closed_loop.ready = true`
- `ret_mean >= 0.80`
- `score_mean >= 150.0`
- `slot_match_mean >= 0.45`
- `rule_apply_error_mean <= 0.01`

Reference baseline:

- executable baseline: `logdir/verify_exec_v5_30k_b4_alien_2bc36b5`
- rollout baseline: `logdir/verify_rollout_v2_30k_b4_alien_304f8ba`

## Immediate Execution Order

The current implementation order is:

1. freshness-aware retrieval support
2. split rollout readiness into two-step and long-horizon
3. light `four_step_apply` training
4. tighter closed-loop and monitor summaries

Planner, actor/value integration, and non-Atari expansion remain explicitly deferred until this four-step loop is stable.
