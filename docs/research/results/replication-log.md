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

## Stage B1 results (2026-11-18, cluster L40S, 50 episodes × 5 tasks, final step)

All 24 rows `done`, no errors in the SLURM logs. Medium seeds 1–2 of FBπS−πh / FBπS are the I1 (b)
runs. Std is over seeds (ddof=1); paper values are Table 6 (5 seeds).

| Env | Method | Seeds 0 / 1 / 2 | Ours | Paper | Δ | Flag (\|Δ\| > 2σ+5) |
|---|---|---|---|---|---|---|
| medium | FB | 44.4 / 54.0 / 58.8 | 52.4 ± 7.3 | 47 ± 1 | +5.4 | no (thr 7) |
| medium | FBπS−πh | 60.0 / 74.0 / 69.6 | 67.9 ± 7.2 | 70 ± 4 | −2.1 | no |
| medium | FB+πh | 58.0 / 65.2 / 53.6 | 58.9 ± 5.9 | 50 ± 12 | +8.9 | no |
| medium | FBπS | 82.4 / 84.4 / 82.8 | 83.2 ± 1.1 | 87 ± 6 | −3.8 | no |
| large | FB | 29.2 / 21.2 / 19.6 | 23.3 ± 5.1 | 21 ± 5 | +2.3 | no |
| large | FBπS−πh | 24.4 / 21.6 / 22.0 | 22.7 ± 1.5 | 37 ± 7 | **−14.3** | no (thr 19) |
| large | FB+πh | 28.8 / 12.0 / 9.2 | 16.7 ± 10.6 | 20 ± 5 | −3.3 | no |
| large | FBπS | 36.4 / 54.0 / 42.0 | 44.1 ± 9.0 | 66 ± 9 | **−21.9** | no (thr 23) |

No cell is flagged, but on Large both FBπS-family numbers are well below the paper (−14 and
−22, each just under the threshold). FBπS−πh on Large is tight (±1.5), so its gap looks systematic,
not seed noise. Per task, all Large FB-family runs score 0–10 % on task 2 (FBπS 28–38 %).
`hyperparameters.sh` uses the same FBπS / FBπS−πh flags on Large as on Medium.

Qualitative claims (spec acceptance criteria):
1. FBπS > FBπS−πh on Medium and Large: **holds** (+15.3 medium, +21.4 large; paper +17, +29).
2. FB+πh ≈ FB: **holds** (+6.5 medium, −6.6 large).
3. FBπS−πh > FB: **holds on Medium** (+15.5), **does not hold on Large** (22.7 vs 23.3; paper 37 vs 21).
4. FBπS within ~1 std of HIQL: **not testable yet**, as HIQL has only Stage A seed 0 on Medium (95.6).
   Against paper HIQL (93 ± 1, 73 ± 6) it would fail on both (83.2, 44.1).

Seed vs machine (I1 caveat): cluster medium seed 0 gives FBπS−πh 60.0 (workstation 61.2) and
FBπS **82.4** (workstation 58.4). So seed 0 is not a bad seed. The workstation FBπS result came from
a different base checkpoint (same seed, same base score, different weights) on a different GPU;
one sample cannot tell which mattered. FB 44.4 vs 49.6 and FB+πh 58.0 vs 29.2 across machines.

Wall time (L40S): FB 48–55 min, FBπS−πh 63–71 min, FB+πh 26–31 min, FBπS 21–24 min.
B1 total ≈ 17 GPU-h (spec estimate 45).

### I1 (c) — Large learning curves from B1 checkpoints (2026-11-19)

`scripts/eval_checkpoint.py`, 50 episodes × 5 tasks, eval seed 0, cluster job 30546379 (ad-hoc
sbatch `slurm_logs/hfb-curves-large.sbatch`, not a registry row: eval-only, no training). Final
checkpoints match each run's `eval.csv` within noise (e.g. FBπS 36.8 / 54.4 / 44.4 vs 36.4 / 54.0 / 42.0).

| Step (k) | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| FBπS−πh sd0 | 20.8 | 18.4 | 36.4 | 23.2 | 14.0 | 11.2 | 17.2 | 26.4 | 17.2 | 23.2 |
| FBπS−πh sd1 | 8.4 | 25.2 | 26.4 | 51.2 | 40.8 | 36.4 | 41.6 | 26.8 | 17.2 | 24.4 |
| FBπS−πh sd2 | 15.2 | 12.4 | 26.8 | 28.4 | 22.4 | 26.8 | 17.6 | 14.8 | 15.6 | 20.0 |
| **FBπS−πh mean** | 14.8 | 18.7 | 29.9 | **34.3** | 25.7 | 24.8 | 25.5 | 22.7 | 16.7 | 22.5 |
| FBπS sd0 | 32.4 | 30.4 | 39.2 | 28.8 | 36.8 | | | | | |
| FBπS sd1 | 55.2 | 53.2 | 47.6 | 54.4 | 54.4 | | | | | |
| FBπS sd2 | 42.4 | 38.8 | 41.6 | 42.0 | 44.4 | | | | | |
| **FBπS mean** | 43.3 | 40.8 | 42.8 | 41.7 | 45.2 | | | | | |

- FBπS−πh on Large peaks at 300–400k (mean 34.3 at 400k, near the paper's 37 ± 7) and then
  declines to about 22 by 1M. Single checkpoints swing by 10–25 points.
- FBπS on Large is flat from 100k on (41–45): the high level converges fast, and its level is set
  by the base it sits on (the 1M base, which is past the base's peak). No later decay.
- So on Large our gap is a late-training decline of the FBπS−πh representation, not a failure of
  the high-level stage. Whether the paper's base runs also decline (and their 37 is just a
  better-behaved run set) cannot be told from the paper.

## I2 — seed-0 cross test: base checkpoint vs machine (2026-11-19)

FBπS medium seed 0 trained on top of the **workstation** FBπS−πh base (`A-fbpis_nonh-medium-sd0-r1`,
61.2 %), once on the cluster (byte copy of the checkpoint, md5-checked) and once again on the
workstation. Registry stages `I2a` / `I2b`; the base rows there point at existing checkpoints
(not retrained). Run group `xtest_I2{a,b}_fbpis_medium`, so they stay out of the main tables.

| Run | Base | Machine | Overall | Task 1–5 |
|---|---|---|---|---|
| A-fbpis-medium-sd0-r1 | workstation base | 3090 | 58.4 | 66 / 84 / 32 / 42 / 68 |
| I2b-fbpis-medium-sd0 | workstation base | 3090 | 62.0 | 62 / 90 / 44 / 48 / 66 |
| I2a-fbpis-medium-sd0 | workstation base | L40S | 62.0 | 60 / 90 / 40 / 48 / 72 |
| B1-fbpis-medium-sd0 | cluster base (60.0 %) | L40S | 82.4 | 94 / 86 / 60 / 82 / 90 |

**The base checkpoint decides it, not the machine.** The workstation base gives 58–62 on either GPU.
The cluster base, with the same seed and the same flat success (60.0 vs 61.2), gives 82.4. A
repeat on the same machine and base moves by only 3.6 points. So the FBπS−πh success rate does
not predict how well a high level can use that base. Combined with I1 (c), the quality of the
1M base checkpoint is the main source of FBπS variance.

## Stage B2 results — 5 seeds (2026-11-19, cluster L40S)

All 26 rows `done`, no errors. Medium/Large × seeds 0–4: seeds 0–2 from B1, 3–4 from B2; HIQL
seeds 0–4 all from B2 (the Stage A workstation HIQL run, 95.6, is not included).

| Env | Method | Seeds 0 / 1 / 2 / 3 / 4 | Ours | Paper | Δ | Flag (\|Δ\| > 2σ+5) |
|---|---|---|---|---|---|---|
| medium | HIQL | 96.4 / 96.4 / 95.2 / 95.6 / 92.4 | 95.2 ± 1.6 | 93 ± 1 | +2.2 | no (thr 7) |
| medium | FB | 44.4 / 54.0 / 58.8 / 48.0 / 45.6 | 50.2 ± 6.1 | 47 ± 1 | +3.2 | no (thr 7) |
| medium | FBπS−πh | 60.0 / 74.0 / 69.6 / 70.8 / 77.2 | 70.3 ± 6.5 | 70 ± 4 | +0.3 | no |
| medium | FB+πh | 58.0 / 65.2 / 53.6 / 64.0 / 57.2 | 59.6 ± 4.9 | 50 ± 12 | +9.6 | no |
| medium | FBπS | 82.4 / 84.4 / 82.8 / 86.8 / 73.6 | 82.0 ± 5.0 | 87 ± 6 | −5.0 | no |
| large | HIQL | 75.2 / 75.6 / 63.2 / 67.2 / 74.0 | 71.0 ± 5.5 | 73 ± 6 | −2.0 | no |
| large | FB | 29.2 / 21.2 / 19.6 / 21.2 / 24.0 | 23.0 ± 3.8 | 21 ± 5 | +2.0 | no |
| large | FBπS−πh | 24.4 / 21.6 / 22.0 / 28.0 / 37.2 | 26.6 ± 6.4 | 37 ± 7 | −10.4 | no (thr 19) |
| large | FB+πh | 28.8 / 12.0 / 9.2 / 23.2 / 12.8 | 17.2 ± 8.4 | 20 ± 5 | −2.8 | no |
| large | FBπS | 36.4 / 54.0 / 42.0 / 56.4 / 47.2 | 47.2 ± 8.3 | 66 ± 9 | **−18.8** | no (thr 23) |

Qualitative claims at 5 seeds:
1. FBπS > FBπS−πh: **holds** (+11.7 medium, +20.6 large; paper +17, +29).
2. FB+πh ≈ FB: **holds** (+9.4 medium, −5.8 large; paper +3, −1).
3. FBπS−πh > FB: **holds on Medium** (+20.1); **marginal on Large** (+3.6, within one std of
   either; paper +16).
4. FBπS within ~1 std of HIQL: **does not hold** — Medium 82.0 ± 5.0 vs 95.2 ± 1.6 (−13),
   Large 47.2 ± 8.3 vs 71.0 ± 5.5 (−24). In the paper the gaps are −6 and −7. HIQL itself
   replicates closely; the gap comes from FBπS being lower, mostly on Large (see I1 (c)).

Wall time (L40S): HIQL 47–50 min, FB 46–55, FBπS−πh 62–71, FB+πh 24–31, FBπS 21–24.
B2 ≈ 19 GPU-h; B1 + B2 ≈ 36 GPU-h.

## Aggregation (2026-11-19)

`scripts/aggregate.py` writes `replication.md` (tables between AUTO markers; the summary and
discrepancy text outside them is hand-written). Its claim rules are stricter than the wording
used above: claim 2 on Medium (FB+πh +9.4 over FB, larger std 6.1) now counts as **fails** under
"|Δ| ≤ larger std", where the B2 entry said "holds".
