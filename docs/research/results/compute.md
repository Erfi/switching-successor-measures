# Compute report (spec 01 Step 0) — 2026-11-18

## Workstation
- Ubuntu 24.04.5, 1× RTX 3090 24 GB, driver 595.99.02 (shared: ~1.7 GB / 30% util in use by other work at check time).
- uv 0.11.28; `.venv` Python 3.11.15. conda 24.11.1 present but unused.
- JAX 0.10.2 sees `CudaDevice(id=0)`; headless MuJoCo (EGL) creates `antmaze-medium-navigate-v0`.
- Datasets: `~/.ogbench/data`. Disk: 375 GB free on `/`.

## Cluster (`ssh kislogin1` → login node `kis3bat1`, Uni Freiburg KI-Cluster)
- Ubuntu 24.04.5. No conda, no modules; uv 0.12.18 in `~/.local/bin`; `.venv` Python 3.11.16.
- Repo: `~/dev_hfb/switching-successor-measures` → `/work/dlc2workfs2/basiri-dev_hfb/switching-successor-measures` (NFS, 419 TB free).
- Home (`/home/basiri`, NFS): 43 GB free. Datasets: `~/datasets/ogbench` (medium, large, giant, teleport + `-val`).
- Internet: through `/etc/profile.d/proxy.sh` (not sourced by non-interactive ssh; must be sourced in job scripts).
- GitHub ssh auth works from the login node.
- Associations: accounts `ais`, `ais-dlc`, `nr-dlc2`, QOS `normal`; jobs submit without `--account`.
- GPU partitions (all 1-day max wall time, `test*` 1 h 5 min):
  - `nrdlc2_gpu-l40s` / `alldlc2_gpu-l40s` / `testdlc2_gpu-l40s`: 30 nodes × 8 L40S, 128 CPUs, 1 TB RAM.
  - `nrdlc2_gpu-h200` / `alldlc2_gpu-h200` / `testdlc2_gpu-h200`: 8 nodes × 8 H200.
  - `alldlc2_gpu-l40s` QOS caps: 100 GPUs, 800 CPUs per user.
- **Used:** `nrdlc2_gpu-l40s` (Erfan's choice).

## Throughput / packing
(to be filled from `scripts/bench_packing.py`)
