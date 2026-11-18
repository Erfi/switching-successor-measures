"""Measure training throughput with 1, 2, 3 concurrent runs on one GPU (spec 01 Step 3, GPU packing).

    python scripts/bench_packing.py --machine=workstation --method=fb --concurrency=1,2,3

Throwaway timing runs (no registry rows): they write to exp_logs_bench/ and are not results.
Throughput comes from `time/epoch_time` (seconds per step) in each run's train.csv, skipping the
first logged interval (JIT compile + the step-1 evaluation).
"""
import argparse
import glob
import os
import shlex
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import logs, matrix  # noqa: E402

REPO_ROOT = matrix.REPO_ROOT


def steps_per_second(run_dir):
    times = [float(r['time/epoch_time']) for r in logs.read_rows(os.path.join(run_dir, 'train.csv'))][1:]
    return 1.0 / statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--machine', required=True, choices=list(matrix.DATASET_DIRS))
    p.add_argument('--method', default='fb', choices=[m for m, v in matrix.METHODS.items() if v['base'] is None])
    p.add_argument('--env', default='medium')
    p.add_argument('--concurrency', default='1,2,3')
    p.add_argument('--steps', type=int, default=20_000)
    args = p.parse_args()

    base = matrix.build_args(args.method, args.env, 0, 'smoke', args.machine)
    base = [a for a in base if not a.startswith(('--save_dir=', '--wandb_run_group='))]
    base += [f'--train_steps={args.steps}', '--log_interval=1000', f'--eval_interval={10 * args.steps}',
             f'--save_interval={10 * args.steps}', '--save_dir=exp_logs_bench']
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false')
    results = []
    for k in [int(c) for c in args.concurrency.split(',')]:
        groups = [f'bench_{args.method}_{args.machine}_c{k}_{i}' for i in range(k)]
        procs = []
        for i, g in enumerate(groups):
            cmd = [sys.executable, 'main.py'] + base + [f'--wandb_run_group={g}', f'--seed={i}']
            print(' '.join(shlex.quote(c) for c in cmd), flush=True)
            procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        codes = [proc.wait() for proc in procs]
        if any(codes):
            sys.exit(f'concurrency {k}: exit codes {codes}')
        per_run = [steps_per_second(sorted(glob.glob(os.path.join(REPO_ROOT, 'exp_logs_bench', g, 'sd*')))[-1])
                   for g in groups]
        results.append((k, per_run))
        print(f'concurrency={k}: per-run steps/s {[round(x, 1) for x in per_run]}, total {sum(per_run):.1f}',
              flush=True)

    one = sum(results[0][1])
    print('\nconcurrency | per-run steps/s | total steps/s | speedup vs 1 | hours per 1M steps (per run)')
    for k, per_run in results:
        mean = statistics.mean(per_run)
        print(f'{k} | {mean:.1f} | {sum(per_run):.1f} | {sum(per_run) / one:.2f}x | {1e6 / mean / 3600:.2f}')


if __name__ == '__main__':
    main()
