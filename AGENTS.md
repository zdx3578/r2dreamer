# AGENTS.md

## Experiment Ops Defaults

- The three experiment machines in this workflow are `2080`, `2070`, and `87`.
- `2070` refers to the local machine in the current workspace, not an SSH host alias.
- SSH experiment hosts are defined in the local SSH config. The canonical remote host aliases in this repo are `2080` and `87`.
- Prefer `2080` and local `2070` for active GPU experiments. Use `87` as extra capacity or fallback.
- Remote code sync is git-only. Do not use `scp` or manual file copies to update experiment machines.

## Remote Update Flow

When a run depends on local code changes, use this sequence:

1. Finish the code changes locally.
2. Review the diff and commit the intended changes.
3. Push to `origin/main`.
4. On the experiment machines, run `git pull --ff-only`.
5. Launch runs only after the machines are on the intended `HEAD`.

If a user asks for remote experiments and the code is not yet on the remote
machine, the expected path is `commit -> push -> git pull`, not ad hoc copying.

## Current Run Profile

Unless the user explicitly overrides it, treat this as the current canonical
throughput profile for structured Alien reruns that need to stay comparable to
the recent routing and weak2 experiments:

- `MAX_PARALLEL=2`
- `batch_size=16`
- `batch_length=64`
- `env.env_num=4`
- `env.train_ratio=128`

Do not silently fall back to older conservative script defaults such as
`batch_size=4`, `batch_length=32`, `env.env_num=1`, `env.train_ratio=32` when
the task is a continuation of the current rerun track.

## Run Reporting

When starting, restarting, or summarizing runs, report:

- host machine
- absolute logdir
- git `HEAD`
- whether the canonical throughput profile above was used
