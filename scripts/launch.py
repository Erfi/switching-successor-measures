"""Register and launch a stage of the run matrix (spec 01 Steps 3 and 5).

Examples (from the repo root, inside the project venv):
    python scripts/launch.py --stage=smoke --backend=local --max_parallel=2
    python scripts/launch.py --stage=A --backend=local --dry_run
    python scripts/launch.py --stage=B1 --backend=slurm --time=08:00:00

Every run becomes a registry row first; execution always goes through scripts/run_one.py.
Rows that already exist and are not `failed` are skipped; failed ones are re-registered as
`<run_id>-r<n>` only with --rerun.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import matrix, registry  # noqa: E402

REPO_ROOT = registry.REPO_ROOT
SLURM_LOG_DIR = os.path.join(REPO_ROOT, 'slurm_logs')
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last}
#SBATCH --output={log_dir}/%x-%A_%a.out
#SBATCH --job-name={job_name}
set -euo pipefail
[ -f /etc/profile.d/proxy.sh ] && source /etc/profile.d/proxy.sh
cd {repo}
RUN_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {run_list})
echo "task $SLURM_ARRAY_TASK_ID -> $RUN_ID on $(hostname)"
nvidia-smi -L
exec .venv/bin/python scripts/run_one.py --run_id="$RUN_ID"
"""


def tracked_changes():
    out = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no', '--', '.', ':!scripts/registry.csv'],
        cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return out


def register(stage, runs, machine, rerun, dry_run):
    """Ensure a registry row per run. Returns run_ids to execute, in order (bases first)."""
    table = matrix.parse_hyperparameters()
    existing = {r['run_id']: r for r in registry.read_all()}
    to_run = []
    for method, env, seed in runs:
        base_id = matrix.base_run_id(stage, method, env, seed)
        attempts = [r for rid, r in existing.items() if rid == base_id or rid.startswith(base_id + '-r')]
        latest = attempts[-1] if attempts else None
        if latest is not None and latest['status'] != 'failed':
            if latest['status'] == 'planned':
                to_run.append(latest['run_id'])  # registered earlier but never started
            else:
                print(f'skip {latest["run_id"]}: status={latest["status"]}')
            continue
        if latest is not None and not rerun:
            print(f'skip {latest["run_id"]}: failed (pass --rerun to register a new attempt)')
            continue
        run_id = registry.next_run_id(base_id)
        command = matrix.build_command(method, env, seed, stage, machine, table)
        print(f'{run_id}: {command}')
        if not dry_run:
            registry.add(dict(run_id=run_id, method=method, env=env, seed=seed, stage=stage,
                              command=command, status='planned', machine=machine))
        to_run.append(run_id)
    return to_run


def run_local(run_ids, max_parallel, log_dir):
    """Queue on this machine: <= max_parallel run_one processes; hierarchical runs wait for their base."""
    os.makedirs(log_dir, exist_ok=True)
    pending = list(run_ids)
    active = {}
    while pending or active:
        for rid, proc in list(active.items()):
            if proc.poll() is not None:
                print(f'[{registry.now()}] finished {rid}: exit {proc.returncode}', flush=True)
                del active[rid]
        for rid in list(pending):
            if len(active) >= max_parallel:
                break
            state = _dependency_state(registry.get(rid))
            if state == 'wait':
                continue
            pending.remove(rid)
            if state == 'blocked':
                registry.update(rid, status='failed', finished_at=registry.now(), notes='base run failed')
                print(f'[{registry.now()}] {rid}: base run failed, not started', flush=True)
                continue
            log_file = os.path.join(log_dir, f'{rid}.log')
            active[rid] = subprocess.Popen(
                [sys.executable, 'scripts/run_one.py', f'--run_id={rid}', f'--log_file={log_file}'], cwd=REPO_ROOT)
            print(f'[{registry.now()}] started {rid} (log {log_file})', flush=True)
        time.sleep(10)
    for rid in run_ids:
        r = registry.get(rid)
        print(f'{rid}: {r["status"]} success={r["final_success"]} {r["notes"]}')


def _dependency_state(row):
    """'ready', 'wait' (base not finished yet) or 'blocked' (base failed or missing)."""
    base_method = matrix.METHODS[row['method']]['base']
    if base_method is None:
        return 'ready'
    bases = registry.find(stage=row['stage'], method=base_method, env=row['env'], seed=row['seed'])
    if not bases:
        return 'blocked'
    status = bases[-1]['status']
    return 'ready' if status == 'done' else 'blocked' if status == 'failed' else 'wait'


def run_slurm(stage, run_ids, args):
    """One array job per (method, env); hierarchical arrays depend element-wise on their base array."""
    os.makedirs(SLURM_LOG_DIR, exist_ok=True)
    groups = OrderedDict()
    for rid in run_ids:
        r = registry.get(rid)
        groups.setdefault((r['method'], r['env']), []).append(r)
    submitted = {}
    for (method, env), rows in groups.items():
        rows.sort(key=lambda r: int(r['seed']))
        name = f'hfb-{stage}-{method}-{env}'
        run_list = os.path.join(SLURM_LOG_DIR, f'{name}.runs')
        script = os.path.join(SLURM_LOG_DIR, f'{name}.sbatch')
        with open(run_list, 'w') as f:
            f.write('\n'.join(r['run_id'] for r in rows) + '\n')
        with open(script, 'w') as f:
            f.write(SBATCH_TEMPLATE.format(
                partition=args.partition, cpus=args.cpus, mem=args.mem, time=args.time, last=len(rows) - 1,
                log_dir=SLURM_LOG_DIR, job_name=name, repo=REPO_ROOT, run_list=run_list))
        cmd = ['sbatch', '--parsable']
        base_method = matrix.METHODS[method]['base']
        if base_method is not None and (base_method, env) in submitted:
            base_job, base_seeds = submitted[(base_method, env)]
            seeds = [int(r['seed']) for r in rows]
            # aftercorr pairs task i with base task i, so the seed order must match exactly.
            dep = 'aftercorr' if seeds == base_seeds else 'afterok'
            cmd.append(f'--dependency={dep}:{base_job}')
        cmd.append(script)
        print(' '.join(cmd))
        if args.dry_run:
            submitted[(method, env)] = ('DRYRUN', [int(r['seed']) for r in rows])
            continue
        job = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]
        submitted[(method, env)] = (job, [int(r['seed']) for r in rows])
        for i, r in enumerate(rows):
            registry.update(r['run_id'], status='queued', job_id=f'{job}_{i}')
        print(f'  -> job {job}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=list(matrix.STAGES))
    p.add_argument('--backend', required=True, choices=['local', 'slurm'])
    p.add_argument('--methods', default=None, help='Comma-separated subset of the stage methods.')
    p.add_argument('--envs', default=None, help='Comma-separated subset of the stage envs.')
    p.add_argument('--seeds', default=None, help='Comma-separated subset of the stage seeds.')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--rerun', action='store_true', help='Register new attempts for failed runs.')
    p.add_argument('--allow_dirty', action='store_true', help='Allow uncommitted tracked changes (non-smoke).')
    p.add_argument('--max_parallel', type=int, default=1, help='local: concurrent runs on the GPU.')
    p.add_argument('--partition', default='nrdlc2_gpu-l40s')
    p.add_argument('--time', default='08:00:00')
    p.add_argument('--cpus', type=int, default=8)
    p.add_argument('--mem', default='32G')
    args = p.parse_args()

    if not matrix.is_smoke(args.stage) and not args.allow_dirty and tracked_changes():
        sys.exit('tracked files have uncommitted changes; commit first (or --allow_dirty):\n' + tracked_changes())

    split = lambda s: s.split(',') if s else None  # noqa: E731
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None
    runs = matrix.stage_runs(args.stage, split(args.methods), split(args.envs), seeds)
    machine = matrix.BACKEND_MACHINE[args.backend]
    run_ids = register(args.stage, runs, machine, args.rerun, args.dry_run)
    if args.dry_run or not run_ids:
        return
    if args.backend == 'local':
        log_dir = os.path.join(REPO_ROOT, matrix.save_dir_of(args.stage), 'launch_logs')
        run_local(run_ids, args.max_parallel, log_dir)
    else:
        run_slurm(args.stage, run_ids, args)


if __name__ == '__main__':
    main()
