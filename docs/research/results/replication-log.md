# Replication log (spec 01)

Running record of deviations from the paper / `hyperparameters.sh` / spec 01, blockers, and
surprises. Newest entries at the bottom of each section.

## Deviations

| # | Date | What | Why |
|---|---|---|---|
| D1 | 2026-11-18 | Python env via **uv** (`.venv`, Python 3.11.15 workstation / 3.11.16 cluster) instead of conda (spec Step 1). | Cluster has no conda; Erfan approved. |
| D2 | 2026-11-18 | `main.py`: new flag `--dataset_dir` (default `~/.ogbench/data` = OGBench default), passed to `make_env_and_datasets`. | Spec Step 1. Default = upstream behaviour. |
| D3 | 2026-11-18 | `main.py`: new flag `--wandb_project` (default `FB pi-Switch` = upstream). Runs use `HFB`. | Erfan's request (separate wandb project). Default = upstream behaviour. |
| D4 | 2026-11-18 | Dataset dir differs per machine: workstation `~/.ogbench/data`, cluster `~/datasets/ogbench`. | Erfan's instruction; no shared FS. Same OGBench files. |
| D5 | 2026-11-18 | Package versions are newer than upstream's: `requirements.txt` uses `>=` for jax/flax/etc., so we got jax 0.10.2, flax 0.12.8, numpy 2.4.6, mujoco 4.0.1, optax 0.2.9, distrax 0.1.9 (ogbench 1.1.4, gymnasium 0.29.1 pinned as upstream). Full lists: `env-freeze-{workstation,cluster}.txt` (identical). | Upstream did not pin. **Risk:** numerical/behavioural differences vs the paper's runs; check Stage A sanity bands. |
| D6 | 2026-11-18 | `--eval_interval=<train_steps>` (eval only at step 1 and at the end), `--eval_episodes=50`, `--video_episodes=0`. | Spec Step 4. |
| D7 | 2026-11-18 | Giant hierarchical runs (FB+πh, FBπS) **without** `--agent.discount=0.995`, as in `hyperparameters.sh`, although the paper says γ=0.995 for Giant. | Spec Step 4: replicate the script as published. |
| D8 | 2026-11-18 | GPU-packing benchmark runs (`scripts/bench_packing.py`, `exp_logs_bench/`) are not in the registry. | Timing-only throwaway runs, not results. |

## Blockers / bugs

- **B1 (2026-11-18): wandb init crashes with the installed wandb.** `utils/log_utils.py:83` passes
  `wandb.Settings(start_method='thread')`; `requirements.txt` has `wandb>=0.19.11`, which resolved
  to wandb 0.31.0, and that rejects the field (`pydantic ValidationError: extra_forbidden`). Every
  run with `--enable_wandb=1` dies at startup. Smoke tests missed it because they use
  `--enable_wandb=0`. Stage A rows `A-*-medium-sd0` failed in about 10 s each (hierarchical ones
  never started). Waiting for Erfan's decision before any fix.

## Notes

- `restore_agent` (`utils/flax_utils.py`) globs its path, but `load_agent_from_frozen` joins
  `frozen_path` with `flags.json` without globbing, so `frozen_path` must be the exact run dir.
  `scripts/run_one.py` fills it with the absolute run dir of the base run (same stage, env, seed).
- `train.csv` is not valid CSV for FB-family agents: `CsvLogger` (`utils/log_utils.py`) writes
  values with `str()`, and `training/actor/std` is a matrix, so its numpy repr spans many lines.
  Not patched (upstream code). `scripts/logs.py:read_rows` rejoins lines inside `[...]`; use it
  for every read of `train.csv` / `eval.csv`.
- Smoke tests (2026-11-18): all 7 agents pass on the workstation (medium, sd0, 2k steps; no NaN,
  eval.csv and `params_2000.pkl` written). For FB+πh and FBπS, the actor, F and B params in the
  2k-step checkpoint are bit-identical to the base checkpoint, so the frozen modules load and stay
  frozen. Cluster smoke (FB → FB+πh, `testdlc2_gpu-l40s`): pass; L40S visible, no dataset
  download triggered, registry updated from compute nodes, `aftercorr` dependency works.
