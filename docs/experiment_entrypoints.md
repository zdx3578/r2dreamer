# Experiment Entrypoints

This note records the current training entrypoints used for the Atari structured runs and the base 50k benchmark.

## Environment

- Python `>= 3.12` is required. `train.py` calls `tools.require_python()` on startup and will fail fast on older interpreters.
- The repository has a single supported command entrypoint:

```bash
./.venv/bin/python train.py ...
```

- `./.venv` should be a symlink or virtualenv directory pointing at the one Python 3.12 environment you want this checkout to use.
- For this repository, do not mix `python`, `python3`, `conda activate`, and multiple environment roots in the same checkout.
- If you need `pip`, `tensorboard`, or `pre-commit`, invoke them through `./.venv/bin/...` as well.
- The benchmark scripts assume they are launched from the repository root.
- Atari 50k runs are typically launched with `device=cuda:0`, `buffer.device=cuda:0`, and `buffer.storage_device=cpu`.

## Structured Training

The current structured Atari entrypoint is:

```bash
./.venv/bin/python train.py \
  env=atari100k \
  +exp=phase2_structured \
  env.task=atari_alien \
  env.steps=50000 \
  env.train_ratio=32 \
  batch_size=4 \
  batch_length=32 \
  trainer.pretrain=0 \
  trainer.eval_episode_num=20 \
  trainer.eval_every=5000 \
  device=cuda:0 \
  buffer.device=cuda:0 \
  buffer.storage_device=cpu \
  model.compile=False \
  logdir=./logdir/structured_alien_seed0 \
  seed=0
```

`+exp=phase2_structured` now also enables the direct spatial supervision path:

- `model.use_structure_decoder=True`
- `model.use_local_decoder=True`
- `model.use_direct_spatial_targets=True`

These add:

- structure decoding from encoder spatial features to `region_map` and `slot_mask`
- local effect decoding from `spatial + z_eff` to `change/roi/local_delta`
- direct spatial-difference targets from frame deltas for `delta_map/delta_obj/reach`

For a very short smoke test:

```bash
./.venv/bin/python train.py \
  env=atari100k \
  +exp=phase2_structured \
  env.task=atari_alien \
  env.steps=32 \
  env.env_num=1 \
  env.eval_episode_num=0 \
  env.train_ratio=1 \
  batch_size=1 \
  batch_length=4 \
  trainer.pretrain=1 \
  trainer.eval_episode_num=0 \
  trainer.eval_every=1000000 \
  trainer.update_log_every=1 \
  device=cpu \
  buffer.device=cpu \
  buffer.storage_device=cpu \
  model.compile=False \
  logdir=/tmp/r2dreamer_structured_smoke \
  seed=0
```

## Base 50k Benchmark

The serial benchmark runner for the plain baselines is:

```bash
./scripts/benchmark_atari_base_50k.sh
```

Default behavior:

- variants: `dreamer r2dreamer`
- seeds: `0 2`
- task: `atari_alien`
- budget: `50000` env steps
- evaluation: `20` episodes every `5000` steps
- execution mode: serial

Useful overrides:

```bash
VARIANTS=dreamer ./scripts/benchmark_atari_base_50k.sh
VARIANTS=r2dreamer ./scripts/benchmark_atari_base_50k.sh
SEEDS="0 2" BASE_LOGDIR=./logdir/my_bench ./scripts/benchmark_atari_base_50k.sh
```

Behavior on rerun:

- completed runs are skipped when `latest.pt` exists
- incomplete runs are moved to `*_aborted_YYYYMMDD_HHMMSS`

To run the benchmark and automatically write a comparison summary:

```bash
./scripts/run_atari_base_50k_with_summary.sh
```

This writes:

- `summary.json`
- `summary.md`

under the benchmark `BASE_LOGDIR`.

## Notes

- The benchmark stability fix includes a `tools.Tee` shutdown guard so completed runs do not fail during interpreter exit when `console.log` has already been closed.
