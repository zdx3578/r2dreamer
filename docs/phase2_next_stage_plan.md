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

## Recent Verification Results

The four-step implementation plan has now effectively landed.
Recent verification changed the meaning of "next stage":

- the new `7k` verification already passes `phase2_executable`, `phase2_rollout`, and `atari_closed_loop`
- the new `30k` verification reached the `23k+` region with strong structure metrics before the run exited early
- the active rerun is therefore a benchmark closure task, not a feature implementation task

The most important interpretation point is:

- low `two/four/seven-step apply_error` is good

Why this is good:

- it means the rule path can be applied repeatedly without obvious compounding error
- it means adding light `four_step_apply` training did not destabilize rollout
- it is exactly the signal we wanted from the current `rho`-shadow rollout stage

What it does not prove by itself:

- it does not yet prove task uplift over the Atari baseline
- it does not yet prove full structured or full latent rollout, because the current rollout is still `rho`-only and teacher-forced on the rest of the path

The latest partial `30k` readout before the aborted run was already structurally strong:

- `slot_match_mean ~= 0.509`
- `object_interface_mean ~= 0.614`
- `retrieval_agreement_mean ~= 0.990`
- `rule_apply_error_mean ~= 6.0e-4`
- `two_step_apply_error_mean ~= 7.6e-4`
- `four_step_apply_error_mean ~= 1.26e-3`
- `seven_step_apply_error_mean ~= 2.53e-3`

The remaining uncertainty is task-side, not structure-side:

- `ret_mean` and `score_mean` were still slightly below the executable reference baseline at that point
- the run itself exited before `30k`, so the benchmark is incomplete

## Baseline Hierarchy

To avoid mixing historical anchors with directly comparable structured baselines, use the following baseline tiers.

### Tier 0: Historical Pre-Phase2 Anchor

- commit: `1fadce4`
- meaning: last convenient repository anchor before the executable Phase2 line existed
- use: a coarse "before these Phase2 modifications" reference point

Important limitation:

- this is not a direct executable-or-rollout benchmark baseline
- no dedicated retained `30k` artifact has been identified in the current workspace for this commit

So `1fadce4` should be recorded as the broadest historical baseline, but not used as the primary apples-to-apples benchmark for the current Phase2 reruns.

### Tier 1: Executable Baseline

- commit: `2bc36b5`
- run: `logdir/verify_exec_v5_30k_b4_alien_2bc36b5`
- role: best reference for one-step executable rule quality before the newer rollout-stage upgrades

Key reference values:

- `slot_match_mean ~= 0.5093`
- `object_interface_mean ~= 0.6172`
- `retrieval_agreement_mean ~= 0.9988`
- `rule_apply_error_mean ~= 4.36e-4`
- `ret_mean ~= 1.0186`
- `score_mean ~= 209.1`

### Tier 2: Rollout Baseline

- commit: `304f8ba`
- run: `logdir/verify_rollout_v2_30k_b4_alien_304f8ba`
- role: best direct baseline for the current rollout-stage branch

Key reference values:

- `slot_match_mean ~= 0.5093`
- `object_interface_mean ~= 0.6186`
- `retrieval_agreement_mean ~= 0.9918`
- `rule_apply_error_mean ~= 3.35e-4`
- `two_step_apply_error_mean ~= 4.13e-4`
- `four_step_apply_error_mean ~= 6.54e-4`
- `seven_step_apply_error_mean ~= 1.25e-3`
- `ret_mean ~= 1.0177`
- `score_mean = 216.0`

### Tier 3: Current Candidate Branch

- code line: `304f8ba` baseline plus later freshness, gating, and monitor upgrades through `3a74aef`
- representative rerun: `logdir/verify_rollout_v3_30k_b4_alien_304f8ba_tmux_rerun_20260310_100507`

Current near-`30k` readout:

- `slot_match_mean ~= 0.5096`
- `object_interface_mean ~= 0.6194`
- `retrieval_agreement_mean ~= 0.9911`
- `rule_apply_error_mean ~= 3.42e-4`
- `two_step_apply_error_mean ~= 4.17e-4`
- `four_step_apply_error_mean ~= 6.60e-4`
- `seven_step_apply_error_mean ~= 1.28e-3`
- `ret_mean ~= 0.9954`
- `score_mean = 189.0`
- `score_max = 270.0`

Current interpretation:

- against the rollout baseline, structure is essentially at parity
- against the executable baseline, structure is still healthy but task return is slightly lower
- the main remaining question is task-side value, not structural correctness

## Current Stage Shift

The old Step 1-5 sequence is history.
The four-step implementation plan is now also mostly history.

The current Phase2 question is no longer:

- can Phase2 execute?
- can Phase2 do longer rollout?

The current question is:

- can the new Phase2 stack finish a clean `30k` benchmark reproducibly?
- after a full benchmark, is Atari task performance at parity with or better than the reference baseline?

## Benchmark Closure Plan

### 1. Finish A Clean 30k Rerun

Goal:

- obtain one complete `30k` benchmark with the current freshness-aware, multi-horizon-gated codepath

Why this is first:

- the model-side changes are already informative enough
- without a complete `30k`, the current branch cannot be signed off as either a success or a regression

### 2. Promote Run Completeness To A First-Class Requirement

Goal:

- treat early process exit as a benchmark failure even when the intermediate metrics look good

Implementation direction:

- record and inspect run completion explicitly
- treat truncated `metrics.jsonl` as an incomplete benchmark
- debug runner stability before making more model-side changes if the rerun fails again

### 3. Freeze Core Phase2 Structure Until Benchmark Closure

Goal:

- avoid changing the model while the current benchmark question is still unresolved

This means:

- do not add `seven_step` loss yet
- do not move to full structured rollout yet
- do not reopen planner, actor/value, or ARC3 scope yet

### 4. If Full 30k Is Structure-Good But Task-Weaker, Run A Narrow Task-Uplift Sweep

Goal:

- improve task return only after structural stability is confirmed

Preferred order:

- first tune small scalar knobs such as `four_step_apply` weight, freshness decay, and gate/warmup schedule
- only consider stronger consumers if multiple full `30k` runs still show structural parity but task underperformance

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

Status:

- already passed on the new codepath

### 3. Mid 16k Stability Check

Purpose:

- verify that freshness and rollout stay stable after warmup

Minimum acceptance:

- `phase2_executable.ready = true`
- `phase2_rollout_two_step_ready = true`
- `phase2_rollout_long_ready = true`
- `retrieval_agreement_mean >= 0.90`
- `rule_apply_error_mean <= 0.01`

Status:

- structurally satisfied in the aborted `23k+` run
- still needs one complete run for sign-off

### 4. Full 30k Benchmark

Purpose:

- verify no regression against the current Atari-only Phase2 baseline on a complete run

Minimum acceptance:

- run reaches the intended `30k` range without early exit
- `atari_closed_loop.ready = true`
- `ret_mean >= 0.80`
- `score_mean >= 150.0`
- `slot_match_mean >= 0.45`
- `rule_apply_error_mean <= 0.01`

Reference baseline:

- executable baseline: `logdir/verify_exec_v5_30k_b4_alien_2bc36b5`
- rollout baseline: `logdir/verify_rollout_v2_30k_b4_alien_304f8ba`

### 5. Post-30k Decision Gate

Decision outcomes:

- if structure is healthy and task is at parity or better, freeze this branch as the new Atari-only Phase2 baseline
- if structure is healthy but task is still weaker, move to a narrow task-uplift sweep
- if the run fails again before completion, debug launcher and runtime stability before any further model changes

## Immediate Execution Order

The current execution order is:

1. let the active `30k` rerun finish
2. read final gate results, window means, and baseline deltas
3. decide whether the branch is benchmark-ready or task-uplift-only
4. only then decide whether another code change is justified

Planner, actor/value integration, full latent rollout, and non-Atari expansion remain deferred until benchmark closure is complete.

## Post-30k Follow-Up Candidates

This section is intentionally post-benchmark.
These items should not be treated as immediate work while the active `30k` rerun is still unresolved.

The ordering below reflects the current best integration of:

- the implemented Phase2 codepath
- the recent `7k` and partial `30k` results
- the longer-horizon rollout observations

### Candidate 1: Upgrade `four_step_apply` From Static Weight To True Curriculum

Current state:

- `four_step_apply` is already in training
- the current implementation uses a fixed small weight
- `two/four/seven-step apply_error` being low is a good sign, not a warning sign

Why this is the first candidate:

- the structure path already looks healthy enough to justify a more explicit training schedule
- a curriculum is the cleanest next refinement if the full `30k` ends up structurally stable but task gains remain ambiguous

Preferred direction:

- keep `two_step_apply` as the default rollout driver
- only enable or strengthen `four_step_apply` when `phase2_rollout_two_step_ready` stays stable
- allow automatic fallback toward two-step dominance if long-horizon metrics degrade

This should be treated as the first post-`30k` model-side change if another code round is needed.

### Candidate 2: Promote Baseline-Relative Metrics Into Closed-Loop Sign-Off

Current state:

- monitoring already computes baseline-relative deltas
- readiness gates still primarily answer whether the system is healthy and runnable

Why this matters:

- the next decision is no longer only "can it run?"
- the next decision is "is this branch at parity with or better than the current Atari-only reference?"

Preferred direction:

- keep the current structural readiness checks
- add a separate baseline-relative decision layer for task and rollout value
- avoid turning this into an overly strict gate before one complete `30k` rerun is available

This should be the second post-`30k` refinement if the branch is structurally sound but benchmark value is still unclear.

### Candidate 3: Improve Retrieval Prior Calibration

Current state:

- freshness-aware support is now the right default
- sparse-memory stages may still over-amplify a small number of active cells

Why this is lower priority:

- the current retrieval path is already good enough to produce healthy executable and rollout metrics
- this is a quality and calibration refinement, not the main blocker

Preferred direction:

- weaken prior sharpness when occupied support is still sparse
- consider soft caps or population-aware temperature on retrieval priors
- keep lifetime stats and freshness stats semantically separate

This should only be touched after the benchmark result clearly justifies another refinement pass.

### Candidate 4: Move From `rho`-Only Shadow Rollout Toward A Rule-Assisted Structured Consumer

Current state:

- today’s rollout is still `rho`-only and teacher-forced on the rest of the path
- low long-horizon apply error does not mean full structured rollout has already been achieved

Why this is deferred:

- this is no longer a small calibration step
- it is the first real step toward a more powerful downstream consumer of the rule path

Preferred direction:

- do not jump directly to full latent overwrite
- first test whether rule outputs can improve structured prediction heads
- keep planner, actor/value conditioning, and full latent intervention out of scope until this shadow-consumer stage is justified

This is a valid next-stage direction, but only after benchmark closure and after the lighter follow-up candidates above have been evaluated.

## Integrated Follow-Up Plan

This section folds in the useful parts of later review notes while correcting the parts that no longer match the current codebase.

### Corrections To Outdated Readings

The current repository is not at a "Phase2 supervision heads only" stage.

What is already true in the current code:

- `RuleMemory` exists and is part of the Phase2 path
- retrieval is already executed during Phase2 forward
- `RuleApply` already fuses head prediction and memory retrieval
- `2/4/7-step` `rho` shadow rollout already exists
- freshness-aware support and rollout gating already exist

So the correct description is:

- Phase2 is already an executable shadow rule interface
- it is not yet a full structured rollout or a full latent rollout consumer

### What To Keep From The Replay Critique

The replay critique is directionally useful, but it must be phrased precisely.

What is true now:

- the default Atari structured runs still use slice replay
- there is not yet an event-rich replay sampler
- there is not yet a freshness-aware replay sampler
- there is not yet a rollout-curriculum replay sampler

What is also true now:

- replay is not completely naive
- the code already supports prioritized replay
- replay priority can already use structure-side signals such as `event_target`, `delta_struct`, and reward magnitude

Interpretation:

- replay organization is no longer just a future lever
- after the `current_head / no_prio` reruns, replay has become the current strongest demonstrated instability amplifier
- this still does not mean `no_prio` is the final repository default
- but it does mean `no_prio` is now the most practical development safety line while deterministic collapse and rerun variance are being reduced

### What To Keep From The Phase1A Critique

The strongest useful point from the Phase1A review is that the current `reach` and `goal` heads are still weak structural consumers.

Current limitations:

- `struct_map / struct_obj / struct_global` still behave more like information-preserving bottlenecks than semantic structure supervision
- `reach` is still closer to a latent-change proxy than an action-conditioned map-reachability head
- `goal` is still closer to a short-horizon reward proxy than a strong structure-conditioned progress head

Important caveat:

- this does not mean Phase1A is failing
- it means Phase1A is currently good enough as a warm-start, but not yet a strong structural consumer

### What To Keep From The Phase1B And Binding Critique

The current multi-seed evidence supports the following ordering:

- if `seed2`-style failures persist, keep working on Phase1B / objectification robustness first
- do not expand `BindingHead` categories just because structure is unstable

Why:

- the observed weak seeds are failing on `slot_match` and `object_interface`
- they are not failing because the binding vocabulary is obviously too small
- adding more binding classes now would add complexity without addressing the demonstrated bottleneck

### Updated Priority Order

After the current robustness loop, the preferred next-stage order is:

1. keep a dual-line evaluation structure:
   - development safety line: `no_prio`
   - performance reference line: `current_head`
2. continue Phase1B robustness work until `seed2`-style structure weakness is either fixed or cleanly bounded
3. only then redesign Phase1A `reach` so it consumes stronger action/effect-conditioned structure signals
4. after replay variance is better bounded, revisit replay upgrades such as event-rich or freshness-aware sampling
5. keep `BindingHead` category expansion out of scope unless later evidence shows a real binding-capacity bottleneck

### Suggested Reach Redesign Direction

When Phase1A becomes the next active target, the preferred direction is:

- make `reach` consume `feat`, `action`, `M_t`, and `z_eff`
- optionally add direct conditioning on `delta_map`
- keep targets proxy-based, but move them closer to map-change or controllable-frontier signals instead of raw latent delta magnitude

This preserves the practical no-external-label setting while making the head more action-conditioned and structurally meaningful.

## Current Closure Update (2026-03-12)

The replay reruns and the `seed_5@2080` control experiment changed the practical execution plan.

### What Is Now Fixed For The Current Stage

Treat the following as frozen unless there is a targeted reason to re-open them:

- `phase1a.direct_target_blend = 0.25`
- restored Phase1B robustness settings
- `phase2.match_gate_mode = soft`
- `phase2.four_step_curriculum_warmup_updates = 1500`
- tri-mode eval
- eval-state actor diagnostics
- `model.actor_imagination.mode_mix = 0.0`

### What Is Now The Minimal Experiment Matrix

Do not keep expanding replay knobs.

For benchmark closure and new feature validation, the active pair is now:

- `current_head`
- `no_prio`

And the active seed set remains:

- `seed_3`
- `seed_4`
- `seed_5`

Interpretation:

- `no_prio` is the current development baseline because it more often reduces deterministic collapse and split severity
- `current_head` is retained as a ceiling / regression reference
- `prio_mean`, `prio_lowalpha`, `blend0`, and `slots32` are archived experimental branches for now

### Current Capacity-Line Conclusion

The `map_slots=32 / obj_slots=32` line showed some short-horizon promise at `20k`, but the `50k` three-machine comparison did not support promoting it.

What the current evidence says:

- it can improve some individual runs
- but it also increases `mode_minus_raw` and split sensitivity
- so it should not be the current main branch

If capacity is revisited later, use a smaller next step such as:

- `obj_slots=16`
- or `map_slots=32, obj_slots=16`

and only on top of the `no_prio` development line.

### Immediate Next Functional Target

The next feature target should not be planner work and should not be another replay sweep.

The preferred next step is:

- make the rule path a stronger prediction-side consumer
- let `delta_rule_fused / rho_next_pred` help `delta_map / delta_obj / delta_global`
- keep actor/value conditioning and planner integration out of scope until this is justified

The validated direction is now narrower than the original residual-consumer idea:

- keep the replay-off development line (`no_prio`) as the only primary baseline
- prefer an aux-only rule consumer first
- keep any direct residual injection to `delta_map / delta_obj / delta_global` on hold unless a weaker schedule is shown cleanly
- keep planner, actor/value consumers, and replay changes out of scope for this tranche

### Practical Execution Rule

For any new feature:

1. first run `20k` smoke A/B on `seed_3/4/5`
2. only then escalate to `50k`
3. require that the feature does not degrade:
   - `raw_mode`
   - `sample_minus_mode`
   - deterministic collapse frequency
4. check `current_head` as the ceiling reference before keeping the change

For the first rule-consumer stage, the default smoke matrix is:

1. `20k`
2. `seed_3/4/5`
3. `no_prio` vs `no_prio_rule_consumer`

Current rule-consumer versioning:

- `v1`: full `map/obj/global` residual consumer, `residual_scale=0.1`
  - too aggressive
  - often increased `mode_minus_raw`
- `v2`: full `map/obj/global`, `residual_scale=0.03`, plus `* artifact.gate.detach()`
  - reduced calibration drift
  - still too invasive on local structure channels
- `v3`: `global-only`
  - strong improvement in average deterministic alignment
  - but `seed_4` still failed the clean `20k` control rerun
- `v4`: `global-only + late enable`
  - same residual strength
  - same gate detachment
  - delayed activation by update schedule
  - first validation target is `seed_4 @ 20k`
- `v5`: `global-only + late enable + phase2 gate threshold`
  - keep the v4 late schedule
  - additionally require detached phase2 gate readiness before enabling residuals
  - first validation targets are `seed_1` and `seed_4`
- `v8`: `global-only + sticky gate + latch ramp`
  - intended to keep residuals narrow and late
  - a later bugfix showed the original apparently good `seed_4 @ 20k` result was a false positive caused by residuals not reaching the Phase1A loss path
  - after the wiring fix, real residual injection degraded `seed_4 @ 20k`
- `v9`: `aux-only + consistency`
  - keeps the rule signal on the prediction side without directly rewriting the main residual path
  - currently the only rule-consumer line with repeatable positive short-horizon evidence

Promotion rule:

- do not escalate rule-consumer changes to `50k` until `seed_4` stops showing repeated clean `20k` scheduled-eval regression

### 2026-03-15 Rule-Consumer Validation Update

Recent rule-consumer experiments now support a clearer split between the residual line and the aux-only line.

#### `v8` Residual Line

What changed:

- the residual path bug was fixed so that consumer-updated `effect_out` also updates `structured["dynamic"]["effect_out"]`
- this made the residual consumer finally train the same tensors used by the Phase1A losses

What the fixed `seed_4 @ 20k` rerun showed:

- `no_prio`: `best_eval=215.0`, `last_eval=159.0`, `raw_mode=51.5`, `mode=145.33`
- `v8 fixed`: `best_eval=148.0`, `last_eval=148.0`, `raw_mode=1.5`, `mode=156.67`
- `rule_consumer_global_residual_ratio` rose as high as `0.562`

Current interpretation:

- the earlier apparently strong `v8` result was not trustworthy because the residual branch had not been wired into the main loss path
- once residuals were truly active, direct residual injection proved too invasive
- the residual line is therefore paused rather than promoted

#### `v9` Aux-Only Line

`20k` short-horizon validation on `seed_3/4/5`:

- `no_prio`: `mean_best=256.67`, `mean_last=120.17`
- `v9`: `mean_best=242.5`, `mean_last=240.17`

Interpretation:

- `v9` clearly improved late-stage stability at `20k`
- this was the first rule-consumer line that looked worth carrying to `50k`

Original `50k` validation on `seed_3/4/5`:

- `no_prio`: `mean_best=251.0`, `mean_last=95.8`
- `v9`: `mean_best=278.5`, `mean_last=43.3`

Interpretation:

- original `v9` could raise peak scores
- but it remained too strong at `50k`, especially on `seed_3` and `seed_5`

First weakened `v9` attempt:

- `AUX_LOSS_SCALE=0.005`
- `CONSISTENCY_LOSS_SCALE=0.0025`
- `START_UPDATES=5000`
- `RAMP_UPDATES=2000`
- `LATCH_RAMP_UPDATES=2000`

This attempt was invalid for algorithm judgment because:

- total `50k` updates in the current throughput profile were only about `3063`
- `START_UPDATES=5000` meant the consumer never activated

Current weakened `v9` configuration (`weak2`):

- `AUX_LOSS_SCALE=0.005`
- `CONSISTENCY_LOSS_SCALE=0.0025`
- `START_UPDATES=1500`
- `RAMP_UPDATES=500`
- `LATCH_RAMP_UPDATES=1000`

Current `weak2` readout:

- `seed_3`
  - `no_prio`: `best_eval=317.0`, `last_eval=0.0`, `raw_mode=0.0`, `mode=1.5`
  - `v9 weak2`: `best_eval=224.5`, `last_eval=41.5`, `raw_mode=30.0`, `mode=30.0`
- `seed_5`
  - `no_prio`: `best_eval=111.5`, `last_eval=48.0`, `raw_mode=30.0`, `mode=30.0`
  - `v9 weak2`: `best_eval=263.0`, `last_eval=100.0`, `raw_mode=100.0`, `mode=100.0`

Important validity point:

- `weak2` really did activate the consumer
- `rule_consumer_effective_scale` reached `1.0`
- `rule_consumer_aux_weight` reached `1.0`
- `rule_consumer_global_residual_ratio` reached about `0.058` on `seed_3` and `0.080` on `seed_5`

Current interpretation:

- weakening and delaying `v9` helped
- `seed_5` improved materially versus the original `v9 @ 50k`
- `seed_3` also improved versus the original `v9 @ 50k`, but still did not beat its baseline
- the correct next checkpoint is `seed_4 @ 50k` with the `weak2` schedule

Current decision gate:

- run `no_prio` vs `no_prio_rule_consumer_aux` on `seed_4 @ 50k`
- keep the `weak2` parameters
- only if `seed_4` also stays clean should the branch rerun `seed_3/4/5 @ 50k`
- only after that should `100k` be considered
