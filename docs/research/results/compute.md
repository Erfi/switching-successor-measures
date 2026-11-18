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

## Throughput / packing (`scripts/bench_packing.py`, FB, medium, 20k steps, training only)

| Machine | Concurrent runs | Steps/s per run | Total steps/s | vs 1 run | h per 1M steps per run |
|---|---|---|---|---|---|
| RTX 3090 (shared with another process) | 1 | 316.7 | 316.7 | 1.00× | 0.88 |
| | 2 | 151.5 | 303.0 | 0.96× | 1.83 |
| | 3 | 100.8 | 302.5 | 0.96× | 2.75 |
| L40S (`testdlc2_gpu-l40s`) | 1 | 486.0 | 486.0 | 1.00× | 0.57 |
| | 2 | 312.8 | 625.7 | 1.29× | 0.89 |
| | 3 | 207.7 | 623.1 | 1.28× | 1.34 |

- 3090: packing gives nothing; run Stage A one run at a time.
- L40S: 2 runs per GPU gives 1.29× total throughput, 3 gives no more. Excludes evaluation time
  (two 5×50-episode evals per run), which Stage A will measure.
- **Decision (Erfan, 2026-11-18): no packing.** One run per GPU on both machines.
