# 01 — Replicate the AntMaze results of "Switching Successor Measures" (Stojanovic & Proutiere, 2026)

Status: ready to execute · Owner: Erfan · Executor: Claude Code
Paper: arXiv 2605.13207 · Upstream code: github.com/stestoKTH/switching-successor-measures (commit 22c938a)

## Goal

Reproduce the paper's goal-reaching numbers (Table 1 / Table 6) for the agents we care about,
using the authors' code **unchanged**, so later work (diagnostics, ablations, the two-basis
hierarchical FB) has a trusted baseline.

Agents in scope, in priority order:

| Name in this plan | Agent file | Stage | Steps |
|---|---|---|---|
| FB | `agents/fb.py` | base | 1M |
| FBπS−πh (non-hierarchical) | `agents/fbpiswitch_nonhierarchical.py` | base | 1M |
| HIQL (reference) | `agents/hiql.py` | base | 1M |
| FB+πh | `agents/fb_hierarchical.py` | on top of frozen FB | 500k |
| FBπS | `agents/fbpiswitch.py` | on top of frozen FBπS−πh | 500k |
| ICVF, One-Step FB (optional, Stage C) | `agents/icvf.py`, `agents/onestep_fb.py` | base | 1M |

Out of scope for this spec: any change to agent logic, new diagnostics, the tabular maze, FBB.

## Ground rules

1. **Do not modify** anything under `agents/` or `utils/` and do not change any `get_config()`
   default. Replication means the published code path. If a bug blocks a run, stop, write it up in
   `docs/research/results/replication-log.md`, and ask before patching.
2. New code lives only in `scripts/` (launchers, registry, eval-only, aggregation) and
   `docs/research/results/`.
3. Every run's command line, seed, run directory, git commit, and status goes into the run
   registry (Step 3). Nothing is launched by hand outside the registry.
4. Use the hyperparameters in `hyperparameters.sh` exactly, plus the evaluation flags in Step 4.
   Any deviation must be listed in the replication log with a reason.

## Step 0 — Compute layout

There are two machines, with different jobs:

| Machine | GPU | Used for |
|---|---|---|
| Local workstation | 1× RTX 3090 (24 GB) | Steps 1–2 (environment, smoke tests), Stage A, measuring throughput |
| SLURM cluster | Cluster GPUs | Stages B–D |

Report the following before anything else, for **both** machines:
- OS, `nvidia-smi`, Python and conda versions, free disk under `~/datasets`.
- On the cluster: `sinfo -s`, the GPU partitions and GPU types, the default and maximum wall
  time, the account or QOS to use, whether compute nodes have internet access, and how
  conda/modules are loaded in batch jobs.
- Anything you can't discover (account name, which partition to use, fair-share limits):
  **ask Erfan**. Don't guess.

## Step 1 — Environment and datasets

- Python 3.11 (upstream used 3.11.3), in a fresh env on each machine:
  `conda create -n ssm python=3.11`, then `pip install -r requirements.txt` (pins
  `ogbench==1.1.4` and `gymnasium==0.29.1`).
- Check that `python -c "import jax; print(jax.devices())"` shows a GPU, both locally and
  **inside a SLURM job** (`srun --gres=gpu:1 ...`). Check that MuJoCo works headless. We disable
  videos, but the environment still needs `MUJOCO_GL=egl`, which `main.py` sets.
- Save `pip freeze > docs/research/results/env-freeze-<machine>.txt` on each machine. `main.py`
  monkey-patches `np.in1d = np.isin`, so note the numpy version.

**Datasets go to `~/datasets/ogbench`, not OGBench's default `~/.ogbench/data`.**
- OGBench's `make_env_and_datasets` accepts `dataset_dir`, and the repo's wrapper forwards
  `**kwargs` to it. Add one flag to `main.py`:
  `flags.DEFINE_string('dataset_dir', '~/.ogbench/data', ...)`, and pass
  `dataset_dir=FLAGS.dataset_dir` in the `make_env_and_datasets(...)` call. The default keeps
  upstream behaviour; this is the only change allowed outside `scripts/`. Log it in the
  replication log. Always launch with `--dataset_dir=~/datasets/ogbench`.
- Download once, on the cluster's **login node** (compute nodes may have no internet), with
  `scripts/download_datasets.py`, a thin wrapper around
  `ogbench.download_datasets([...], dataset_dir='~/datasets/ogbench')`. Download the four navigate
  datasets: medium, large, giant, teleport. Each comes with a `-val` file, which is also needed
  because zero-shot latents are inferred from it. Do the same on the workstation.
- Confirm a training job on a compute node starts **without** triggering a download.

## Step 2 — Smoke tests (minutes, not hours)

For each agent in scope, on `ogbench-antmaze-medium-navigate-v0`, seed 0:

```bash
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/<agent>.py \
  --train_steps=2000 --log_interval=500 --eval_interval=2000 --save_interval=2000 \
  --eval_episodes=1 --video_episodes=0 --enable_wandb=0 \
  --dataset_dir=~/datasets/ogbench --save_dir=exp_logs_smoke --wandb_run_group=smoke_<agent>_medium
```

- Hierarchical agents need a frozen base checkpoint. For the smoke test use the 2k-step base run:
  `--agent.frozen_path=<exact base run dir> --agent.restore_epoch=2000`.
  `frozen_path` must be the **exact** run directory (it is joined with `flags.json` without
  globbing); `restore_epoch` must match a saved `params_<epoch>.pkl`.
- Pass criteria: no crash, no NaN in `train.csv`, `eval.csv` written, checkpoint saved,
  hierarchical agent loads the frozen modules.

## Step 3 — Run registry and launcher (`scripts/`)

Run directories are `<save_dir>/<wandb_run_group>/sd<seed>_[s_<SLURM_JOB_ID>.]<timestamp>`, so
paths are not predictable. Build:

- `scripts/registry.csv`: one row per run — `run_id, method, env, seed, stage, depends_on,
  command, run_dir, git_commit, status, final_success`.
- `scripts/launch.py`: reads a stage from the matrix (Step 5), writes the commands, and submits
  them with one of two backends. It fills in `run_dir` after the run starts by globbing
  `<save_dir>/<group>/sd<seed:03d>_*`. The SLURM job ID is part of the directory name, so this is
  unambiguous on the cluster.
  - `--backend=local`: a queue on the 3090 that runs N jobs at once on the same GPU (see
    "GPU packing" below).
  - `--backend=slurm`: generates `sbatch` scripts. Use one array job per (stage, method, env),
    with `--gres=gpu:1`, the partition/account from Step 0, and the env activation. Set wall time
    to 2× the time measured in Stage A. Put logs in `slurm_logs/`.
- **No resume.** `main.py` restarts at step 1 even when given `--restore_path`. Don't use
  preemptible partitions, and size wall times so jobs don't get killed. If a job is killed, rerun
  it from scratch and record that in the registry.
- **Hierarchical dependencies:** submit base runs, then submit each hierarchical array with
  `--dependency=afterok:<base_array_job>` (element-wise via `aftercorr` if the cluster supports
  it). The launcher fills `--agent.frozen_path` from the registry when the hierarchical job
  starts, so the path must be resolved inside the job script, not at submission.
- **GPU packing:** these runs use small MLPs with batch size 1024 and won't saturate a GPU.
  In Stage A, measure throughput (steps/s) with 1, 2 and 3 concurrent runs per GPU, using
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`). If total
  throughput scales, pack runs: several `srun` steps per job on the cluster, several processes on
  the 3090. Report the measured scaling before Stage B.
- Hierarchical runs are submitted only after their base run (same env, **same seed**) has
  `params_1000000.pkl`; they get `--agent.frozen_path=<that run_dir>`.
- Group naming: `--wandb_run_group=<method>_<envshort>` (e.g. `fbpis_nonh_large`).

## Step 4 — Commands

Take every command from `hyperparameters.sh` verbatim, and append these flags to all of them:

```
--seed=<0..4> --eval_episodes=50 --video_episodes=0 --enable_wandb=<0|1> \
--dataset_dir=~/datasets/ogbench \
--save_dir=exp_logs --wandb_run_group=<method>_<env>
```

- The paper evaluates **50 episodes × 5 tasks**, while `main.py` defaults to 20. Use 50.
- Evaluation is expensive (5 tasks × 50 episodes × 1000 steps). To save GPU time, set
  `--eval_interval` equal to `--train_steps`. Evaluation then runs at step 1 and at the end.
  Checkpoints are still saved every 100k, so learning curves can be recomputed later with the
  eval-only script (Step 6).
- **Giant discount, a known discrepancy.** `hyperparameters.sh` passes
  `--agent.discount=0.995` to the Giant *base* runs, but not to the Giant *hierarchical* runs
  (FB+πh, FBπS). The paper says γ = 0.995 for Giant. Replicate the script as published, and log
  this in the replication log. Giant is low priority anyway (Step 5).
- wandb is optional. `train.csv` and `eval.csv` are always written. If wandb is used, the
  project name is hard-coded as `FB pi-Switch`.

## Step 5 — Run matrix, in stages

Don't start a stage until the previous one passes.

| Stage | Where | Envs | Seeds | Methods | Runs | Purpose |
|---|---|---|---|---|---|---|
| A | 3090 | medium | 0 | FB, FBπS−πh, HIQL → then FB+πh, FBπS | 5 | Validate the pipeline end to end, measure time and packing |
| B1 | SLURM | medium, large | 0–2 | FB, FBπS−πh, FB+πh, FBπS | 24 | Core FB-family comparison at 3 seeds |
| B2 | SLURM | medium, large | 3–4 (+ HIQL 0–4) | same 4 + HIQL | 26 | Complete 5 seeds; HIQL as a reference |
| C | SLURM | teleport (+ ICVF, One-Step FB on medium/large, optional) | 0–4 | 5 (+2) | 25 (+20) | Remaining Table 1 columns |
| D | SLURM | giant | 0–4 | FB, FBπS−πh, FB+πh, FBπS | 20 | Needed for the general-reward eval |

**Budget estimate**, to be replaced by Stage A measurements. The paper reports at most 2–3 GPU-h
per 1M-step run on an A40. Hierarchical runs are 500k steps and train only the high-level actor
(plus F′ for FB+πh); assume about 1–1.5 h.
- Stage A: about 10 GPU-h, i.e. overnight on the 3090.
- Stage B1: about 45 GPU-h.
- Stage B2: about 55 GPU-h.
- Giant runs probably take longer, because their episodes are 2000 steps.

Packing (Step 3) may cut wall-clock time 2–3×. Stop after B1 and report: if the four qualitative
claims are already clearly settled at 3 seeds, Erfan decides whether B2 is worth the compute.

**Stage A sanity bands** (seed 0, medium, final success): FBπS−πh ≳ 55, FBπS ≳ 75,
HIQL ≳ 85, FB and FB+πh roughly 30–60. Anything far outside these bands: stop and investigate
before running Stage B.

## Step 6 — Eval-only script (`scripts/eval_checkpoint.py`)

Needed for learning curves and for the general-reward tasks (Table 7) without retraining.

- Inputs: `--run_dir`, `--epoch`, `--complex_task_name` (None or `regions`),
  `--eval_episodes`, `--seed`.
- Read `flags.json` from `run_dir`, recreate the agent with the saved `agent` config, and call
  `restore_agent(agent, run_dir, epoch)`. The saved params of hierarchical agents already contain
  the frozen modules, so do not call `load_agent_from_frozen` again.
- Run the same evaluation path as `main.py`: relabel the validation set → `infer_latent` →
  `evaluate`. Import the existing functions; don't re-implement them. Write a CSV next to the
  checkpoint.
- **Check before trusting it:** for one Stage A run, the eval-only result at the final epoch
  must match that run's own final `eval.csv` within noise.
- General rewards: `--complex_task_name=regions` on Large and Giant, for FB, FB+πh, FBπS−πh and
  FBπS (plus ICVF and One-Step FB if they were trained). HIQL is not evaluated there.

## Step 7 — Aggregation and report

`scripts/aggregate.py` → `docs/research/results/replication.md` containing:

1. **Goal-reaching table:** our mean ± std over seeds next to the paper's Table 6 values (below),
   using final-step `evaluation/overall_success × 100`.
2. **General-reward table:** per-task returns and IQM with stratified-bootstrap 95% CIs, computed
   the way the paper's Appendix C.5 describes (per-task min–max normalization across all methods
   and seeds, then IQM). Compare against Table 7.
3. **Learning curves** from the 100k checkpoints, if the eval-only sweep was run.
4. **Run inventory:** run time per run and failed or re-run jobs.
5. **Discrepancy list:** every deviation from the paper or the script, and anything surprising.

### Paper reference values (Table 6, success %, mean ± std over 5 seeds)

| Method | Medium | Large | Giant | Teleport |
|---|---|---|---|---|
| HIQL | 93 ± 1 | 73 ± 6 | 5 ± 3 | 42 ± 4 |
| ICVF | 46 ± 37 | 28 ± 3 | 0 ± 0 | 23 ± 6 |
| One-Step FB | 35 ± 7 | 23 ± 18 | 0 ± 0 | 8 ± 2 |
| FB | 47 ± 1 | 21 ± 5 | 0 ± 0 | 25 ± 7 |
| FB+πh | 50 ± 12 | 20 ± 5 | 0 ± 0 | 13 ± 7 |
| FBπS | 87 ± 6 | 66 ± 9 | 1 ± 1 | 40 ± 6 |
| FBπS−πh | 70 ± 4 | 37 ± 7 | 1 ± 1 | 29 ± 4 |

Table 7 IQM (general rewards):
- Large: ICVF 0.48, One-Step FB 0.09, FB 0.37, FB+πh 0.24, FBπS 0.67, FBπS−πh 0.35.
- Giant: ICVF 0.43, One-Step FB 0.21, FB 0.33, FB+πh 0.30, FBπS 0.65, FBπS−πh 0.63.

IQM scores aren't comparable in absolute terms unless the same set of methods is normalized
together. Compare *rankings* if the method set differs.

## Acceptance criteria

- Stage B1 done (4 FB-family methods × {medium, large} × 3 seeds) and reported. B2 is run if
  Erfan approves it.
- For each (method, env), flag cases where |ours − paper| > 2 × paper std + 5 points.
- These qualitative claims must hold, or be explicitly reported as not holding:
  1. FBπS > FBπS−πh on Medium and Large: hierarchy helps.
  2. FB+πh ≈ FB: adding a high-level policy to plain FB gives only small gains.
  3. FBπS−πh > FB: the action-free expectile representation helps even without hierarchy.
  4. FBπS is within roughly 1 std of HIQL on Medium and Large.
- `replication.md` and `replication-log.md` are written and committed, along with all scripts.

## Things to watch (from reading the code)

- **No terminal masking anywhere.** The paper notes HIQL comes out lower than OGBench's own
  numbers because of this (93/73/5/42 vs 96/91/65/42). Don't "fix" it.
- **Goal noise is disabled at reset** (`eval_env.unwrapped._add_noise_to_goal = False`). Keep it
  that way.
- **FB-family agents never see the goal state.** Their task latent is inferred from an x-y
  indicator reward (radius 0.5) on the validation set, by `relabel_dataset` → `infer_latent`.
- **Evaluation is deterministic:** `--eval_temperature=0` is the default.
- `restore_epoch` in the hierarchical configs defaults to `1_000_000`. That's right for full
  runs; override it only for smoke tests.
- Seeds pair across stages: a hierarchical run with seed k must freeze the base run with seed k.
