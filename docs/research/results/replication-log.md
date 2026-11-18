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
| D9 | 2026-11-18 | **wandb pinned to 0.19.11** (`scripts/constraints.txt`; `requirements.txt`'s own lower bound). Side effect: protobuf 7.36.2 → 6.33.6. Env freezes updated, still identical on both machines. | Fixes blocker B1 without touching `utils/`. Erfan's choice. |

## Blockers / bugs

- **B1 (2026-11-18): wandb init crashes with the installed wandb.** `utils/log_utils.py:83` passes
  `wandb.Settings(start_method='thread')`; `requirements.txt` has `wandb>=0.19.11`, which resolved
  to wandb 0.31.0, and that rejects the field (`pydantic ValidationError: extra_forbidden`). Every
  run with `--enable_wandb=1` dies at startup. Smoke tests missed it because they use
  `--enable_wandb=0`. Stage A rows `A-*-medium-sd0` failed in about 10 s each (hierarchical ones
  never started). **Resolved** by D9 (pin wandb==0.19.11, Erfan's choice). Checked with a 500-step FB run with wandb on (offline mode). Stage A relaunched as `A-*-r1`.

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

## Investigation I1 — Stage A FBπS below sanity band (2026-11-18)

Stage A (workstation, medium, seed 0, 50 episodes × 5 tasks, final step):

| Run | Overall | Task 1–5 | Paper (Table 6) | Flag (\|Δ\| > 2σ+5) |
|---|---|---|---|---|
| HIQL | 95.6 | 98 / 96 / 96 / 92 / 96 | 93 ± 1 | no |
| FB | 49.6 | 82 / 64 / 28 / 2 / 72 | 47 ± 1 | no |
| FBπS−πh | 61.2 | 66 / 92 / 42 / 28 / 78 | 70 ± 4 | no |
| FB+πh | 29.2 | 0 / 0 / 36 / 52 / 58 | 50 ± 12 | no |
| **FBπS** | **58.4** | 66 / 84 / 32 / 42 / 68 | 87 ± 6 | **yes** |

Wall time on the 3090: base runs 52–58 min each, FB+πh 36 min, FBπS 21 min (500k steps, incl. 2 evals).

Checks (no code changed):
1. **Training curves (FBπS high actor):** no divergence or NaN. The high actor plateaus within
   the first 5k steps: train MSE to target latent 0.76 → 0.71, val MSE 0.74 → 0.75 (slowly
   rising), log-prob flat at about −164, advantage saturation 0.2–0.5%, all samples active.
   FB+πh: advantages are very large (mean ~4·10³, std ~2·10⁴), 5–6% of weights saturate the
   clip at `high_alpha=1e-3`.
2. **Frozen modules:** final checkpoints of FB+πh and FBπS hold actor, F and B bit-identical to
   their base runs' `params_1000000.pkl`; `frozen_path` = exact base run dir, `restore_epoch`=1M.
3. **Hyperparameters vs paper (Tables 3–5, App. C):** all FBπS / FBπS−πh / FB / FB+πh values
   match. Only difference: `orthonorm_coeff` = 1e-3 in the `fbpiswitch.py` config versus 1e-4 in
   Table 4, but the coefficient is unused when only the high actor trains (base used 1e-4).
   Evaluation also matches the paper: a fresh subgoal latent every step, deterministic
   (temperature 0).
4. **Per task:** FBπS beats its base on tasks 2 and 4 (+16, +14) and loses on 1, 3, 5
   (−0, −10, −10); net −3. In the paper the high level adds +17 over the base (70 → 87).
   FB+πh drops to 0 on tasks 1–2 where FB alone gets 82/64.
5. **Versions:** not tested. Paper states no JAX/library versions; ours are much newer (D5).

Open: one seed only. The base FBπS−πh is itself 9 points under the paper mean, and the
high level adds nothing on top of it here.

### I1 (a) — learning curves from checkpoints (`scripts/eval_checkpoint.py`, 50 episodes × 5 tasks)

Eval-only check (spec Step 6): final-checkpoint re-evaluation matches each run's own `eval.csv`
within noise — FBπS 57.6 vs 58.4, FBπS−πh 60.4 vs 61.2, FB+πh 31.2 vs 29.2.

| Step (k) | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| FBπS−πh | 46.4 | 34.4 | 77.6 | 69.6 | 60.0 | 58.4 | 73.2 | 62.0 | 54.4 | 60.4 |
| FBπS (500k on top of FBπS−πh@1M) | 64.8 | 63.2 | 67.2 | 73.6 | 57.6 | | | | | |
| FB+πh (500k on top of FB@1M) | 33.2 | 29.6 | 29.6 | 27.6 | 31.2 | | | | | |

- Checkpoint-to-checkpoint swings (FBπS−πh 54–78 after 300k) are far larger than episode
  sampling noise (≈3 points SE for 250 episodes): the final-step number of a single run is noisy.
- FBπS averages 65.3 over its 5 checkpoints vs 61.7 for FBπS−πh over 600k–1M: the high level
  adds about +3–4 points here, not the paper's +17. FBπS never exceeds 74 at any checkpoint
  (paper 87 ± 6).
- FB+πh is flat around 30 for the whole stage, below FB's 49.6.

### I1 (b) — extra seeds on the cluster (B1 rows, L40S, commit 11fb1f8)

| Run | Seed | Overall | Task 1–5 | Wall time |
|---|---|---|---|---|
| FBπS−πh | 1 | 74.0 | 82 / 90 / 76 / 38 / 84 | 70 min |
| FBπS−πh | 2 | 69.6 | 52 / 92 / 72 / 56 / 76 | 63 min |
| FBπS | 1 | 84.4 | 76 / 92 / 92 / 74 / 88 | 22 min |
| FBπS | 2 | 82.8 | 72 / 90 / 86 / 78 / 88 | 21 min |

Seeds 0–2 (seed 0 = Stage A on the workstation): FBπS−πh 61.2 / 74.0 / 69.6 → mean 68.3 ± 6.5
(paper 70 ± 4); FBπS 58.4 / 84.4 / 82.8 → 75.2 ± 14.6 (paper 87 ± 6). On seeds 1–2 the high
level adds +10.4 and +13.2 over its base, in line with the paper's +17.

**Conclusion:** seed 0 looks like an outlier, not a pipeline bug. Caveat: seed 0 is also the only
run on the workstation (RTX 3090 vs L40S), so seed and machine are confounded; B1's cluster run
of medium seed 0 resolves this.

**Decision (Erfan, 2026-11-18):** treat Stage A seed 0 as an outlier; Stage A passes; continue with B1.
