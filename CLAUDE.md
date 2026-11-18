# CLAUDE.md — switching-successor-measures (Erfan's fork)

## What this repo is
Fork of the official code for "Switching Successor Measures for Hierarchical Zero-shot RL"
(Stojanovic & Proutiere, arXiv 2605.13207). JAX/flax, offline training on OGBench AntMaze.
Agents: FB, FB+πh (`fb_hierarchical.py`), FB π-Switch (`fbpiswitch_nonhierarchical.py` = stage 1,
`fbpiswitch.py` = high-level stage 2), HIQL variants, ICVF, One-Step FB.

## Source of truth
- `docs/research/` holds the design specs and results. Read the spec you're given before acting.
  Specs are numbered (`01-...md`). Results go to `docs/research/results/`.
- Design discussions happen outside this repo. Anything that needs a design decision goes back to
  Erfan as a question; don't decide it in code.

## Rules
- The upstream agent code is the baseline. Don't change `agents/*` or `utils/*`, or any
  `get_config()` default, unless the spec says so. New behaviour goes behind a new config flag
  that defaults to the upstream behaviour.
- New tooling goes in `scripts/`. Keep it importable and reuse `main.py` and `utils/` functions
  rather than copying them.
- Every run goes through the run registry (`scripts/registry.csv`), recording the exact command,
  seed, run dir, and git commit. Launch with `scripts/launch.py`; it registers rows and executes
  them through `scripts/run_one.py`. Run definitions live in `scripts/matrix.py`. The registry is
  shared through git: commit and push it after each stage, and pull before launching elsewhere.
- Hierarchical stages (`fb_hierarchical.py`, `fbpiswitch.py`) need
  `--agent.frozen_path=<exact base run dir>`. Use the same seed as the base run.
- Before any long run: do a 2k-step smoke test with `--enable_wandb=0 --video_episodes=0`.
- Use plan mode first for any multi-file change, and wait for approval.

## Compute and data
- Local workstation: 1× RTX 3090. Use it for smoke tests and small stages only.
- SLURM cluster: use it for all multi-seed sweeps, through `scripts/launch.py --backend=slurm`.
  Never run long jobs on the login node; it's only for downloads and submitting jobs.
- No shared filesystem between the two machines. Code moves only by git push/pull on branch
  `replication`, and only between these two clones (never touch other repos on the cluster):
  - workstation: `~/Dev/switching-successor-measures`
  - cluster: `ssh kislogin1`, repo at `~/dev_hfb/switching-successor-measures` (on NFS `/work`).
- Python env: uv, not conda. `.venv/` (Python 3.11) in the repo root on both machines, created with
  `uv venv --python 3.11 .venv && uv pip install -r requirements.txt -c scripts/constraints.txt`
  (the constraints pin wandb==0.19.11; newer wandb breaks upstream `setup_wandb`). Pinned snapshot of the
  installed versions: `docs/research/results/env-freeze-*.txt` (identical on both machines).
- Datasets (pass `--dataset_dir`; `scripts/matrix.py` picks it per backend):
  - workstation: `~/.ogbench/data`
  - cluster: `~/datasets/ogbench`. Pre-download on the login node.
- Cluster internet (login node over non-interactive ssh, and compute nodes) needs
  `source /etc/profile.d/proxy.sh` first.
- SLURM: partition `nrdlc2_gpu-l40s` (L40S), default account, 1-day wall-time limit.
  Quick tests: `testdlc2_gpu-l40s` (1 h limit).
- wandb: project `HFB` (`--wandb_project=HFB`), default entity. Smoke tests use `--enable_wandb=0`.
- Cluster results come back by rsync of `flags.json`, `train.csv`, `eval.csv` only, into the
  gitignored `exp_logs_cluster/`. Checkpoints stay on the cluster.
- `main.py` has no resume: size wall times generously and avoid preemptible partitions.

## Run conventions
- Run dir: `<save_dir>/<wandb_run_group>/sd<seed:03d>_[s_<jobid>.]<timestamp>/` containing
  `flags.json`, `train.csv`, `eval.csv`, `params_<step>.pkl`.
- Final eval: `--eval_episodes=50`, deterministic (`--eval_temperature=0`, the default).
- Giant: base runs use `--agent.discount=0.995` (see `hyperparameters.sh`).
