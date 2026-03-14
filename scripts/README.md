# Scripts Notes

This directory contains the practical runner stack used for the Phase2
structured Alien experiments.

## Runner Stack

- `run_phase2_structured_alien_repro_one.sh`
  The per-seed training + probe entrypoint.
- `run_phase2_structured_alien_repro_parallel.sh`
  The shared multi-seed launcher used by the matrix wrappers.
- `run_phase2_structured_alien_*_ab.sh`
  Thin experiment-matrix wrappers built on top of the shared launcher.

## Baseline Throughput

The base structured Alien runner still carries the original conservative
per-run defaults from `run_phase2_structured_alien_repro_one.sh`:

- `batch_size=4`
- `batch_length=32`
- `env.env_num=1`
- `env.train_ratio=32`

Those values were chosen for safe 20k/50k reproduction, not for saturating
modern GPUs.

## Current Throughput Profile

For the current follow-up runs we use this higher-throughput profile in the
wrapper scripts:

- `MAX_PARALLEL=2`
- `batch_size=8`
- `batch_length=64`
- `env.env_num=2`
- `env.train_ratio=32`

This profile is now the default in:

- `run_phase2_structured_alien_deterministic_debug_ab.sh`
- `run_phase2_structured_alien_phase1a_followup_ab.sh`
- `run_phase2_structured_alien_deterministic_main_ab.sh`
- `run_phase2_structured_alien_phase1a_main_ab.sh`

Why this profile:

- It raises the GPU-side batch/workload size.
- It raises environment-side data production with `env.env_num=2`.
- It does not change the update/data ratio, so it is less likely to distort
  wall-clock speed or training behavior than jumping straight to
  `env.train_ratio=128`.

## Tuning Guidance

Do not treat `process CPU > 100%` as proof that the machine is CPU-bound.
That only means one training process is using more than one core.

The correct check is machine-level headroom:

- If per-process CPU is high but machine CPU still has large idle headroom,
  the system is not yet CPU-saturated.
- In that situation, increasing `env.env_num` is often the first useful knob,
  because Atari stepping is CPU-side and can feed the GPU better.
- If machine CPU idle becomes low, then `env.env_num` has probably gone too far.

Observed with the current follow-up profile:

- `2070`: GPU about `30%`, machine CPU about `83% idle`
- `2080`: GPU about `59%`, machine CPU about `77% idle`

That means the new profile improved utilization, and both machines still have
CPU headroom left.

## Recommended Knob Order

If you want more throughput without badly changing experiment semantics, tune
in this order:

1. `env.env_num`: `1 -> 2 -> 4`
2. `batch_length`: `32 -> 64`
3. `batch_size`: `4 -> 8 -> 12/16`
4. `MAX_PARALLEL`: increase only if each single run is still too light
5. `env.train_ratio`: keep at `32` unless you explicitly want more optimizer
   work per environment step

## Why Not Jump Straight To train_ratio=128

`env.train_ratio=128` can increase GPU work, but it also increases the number
of optimizer updates per environment step. That often improves utilization at
the cost of wall-clock speed, so it is not the first knob for "finish the
experiment faster".

Use `train_ratio=128` only as a deliberate experiment choice, not as the
default throughput fix.
