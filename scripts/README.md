# Scripts Notes

This directory contains the practical runner stack used for the Phase2
structured Alien experiments.

## Agent Defaults

For new Codex sessions, the repo-level operational defaults now live in
`AGENTS.md` at the repository root.

Use that file for concise, durable rules such as:

- which machines are local vs remote for experiments
- which SSH host aliases are canonical for remote machines
- whether remote sync is git-only
- the current canonical run profile that new sessions should assume

Keep script-specific details and tuning rationale in this `README`, not in
`AGENTS.md`.

The machine-executed defaults for the current structured Alien rerun track now
live in `scripts/structured_alien_defaults.sh`. Update that file when the
canonical runtime profile changes.

## Git Sync Standard

Experiment rollout is git-only. Do not use `scp` to push runner scripts.

Standard flow:

1. Commit local runner or config changes.
2. Push to `origin/main`.
3. Run `scripts/pull_experiment_repos.sh`.
4. Start tmux jobs only after both machines are on the same `HEAD`.

Helper:

- `scripts/pull_experiment_repos.sh`
  Runs `git pull --ff-only` in this repo locally and on the configured remote
  hosts. Default remote host list is `2080`. Override with
  `REMOTE_HOSTS="2080 otherhost"`.

Guardrail:

- `run_phase2_structured_alien_repro_parallel.sh` now refuses to launch if
  `HEAD != origin/main`.
- It also refuses to launch from a dirty worktree unless
  `ALLOW_DIRTY_RUN=1` is explicitly set.

## Runner Stack

- `run_phase2_structured_alien_repro_one.sh`
  The per-seed training + probe entrypoint.
- `run_phase2_structured_alien_repro_parallel.sh`
  The shared multi-seed launcher used by the matrix wrappers.
- `run_phase2_structured_alien_*_ab.sh`
  Thin experiment-matrix wrappers built on top of the shared launcher.

## Shared Runtime Defaults

The base structured Alien runner now sources
`scripts/structured_alien_defaults.sh`, so the machine-executed defaults are:

- `MAX_PARALLEL=2`
- `batch_size=16`
- `batch_length=64`
- `env.env_num=4`
- `env.train_ratio=128`

The older conservative profile
`batch_size=4 / batch_length=32 / env.env_num=1 / env.train_ratio=32`
is now only a historical reference.

## Current Comparability Profile

For the current structured Alien rerun track, the canonical comparability
profile is:

- `MAX_PARALLEL=2`
- `batch_size=16`
- `batch_length=64`
- `env.env_num=4`
- `env.train_ratio=128`

This is the profile to use when continuing the recent routing and weak2
experiments and you want new runs to stay directly comparable with that line.

Important:

- This is an experiment-profile default, not a universal tuning rule.
- Some legacy wrappers still carry lighter defaults or different knob choices.
- When those wrappers are reused for the current rerun track, override them
  explicitly or update the wrapper before launching.

Why this distinction matters:

- The comparability profile preserves continuity with the latest reruns.
- The generic tuning guidance below is still useful when designing a new run
  profile from scratch.

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

## About train_ratio=128

`env.train_ratio=128` increases optimizer work per environment step. That can
be the right choice for a specific comparison track, but it is still not the
first generic knob for "make any run faster".

Use it deliberately:

- use `train_ratio=128` when the current experiment line already depends on it
- do not switch to it silently when you are doing fresh throughput tuning
