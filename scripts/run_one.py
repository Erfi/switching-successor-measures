"""Execute one registry row: resolve the frozen base run, run main.py, record the outcome.

Usage (from the repo root, inside the project venv):
    python scripts/run_one.py --run_id=A-fb-medium-sd0 [--log_file=path]

Used by both launcher backends. Exits non-zero if the run failed, so SLURM dependencies work.
"""
import argparse
import csv
import glob
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import matrix, registry  # noqa: E402

REPO_ROOT = registry.REPO_ROOT


def git_commit():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no', '--', '.', ':!scripts/registry.csv'],
        cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return head + ('-dirty' if dirty else '')


def job_id():
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        return f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
    if 'SLURM_JOB_ID' in os.environ:
        return os.environ['SLURM_JOB_ID']
    return f'pid{os.getpid()}'


def flag_value(args, name):
    return matrix._flag_value(args, name)


def resolve_frozen_path(row, args):
    """Absolute run_dir of the base run (same stage, env, seed), checked for the checkpoint to restore."""
    base_method = matrix.METHODS[row['method']]['base']
    if base_method is None:
        return args
    restore = flag_value(args, '--agent.restore_epoch')
    restore_epoch = int(restore.replace('_', '')) if restore else matrix.DEFAULT_RESTORE_EPOCH
    bases = registry.find(stage=row['stage'], method=base_method, env=row['env'], seed=row['seed'], status='done')
    if not bases:
        raise RuntimeError(f'no finished {base_method} run for stage={row["stage"]} env={row["env"]} '
                           f'seed={row["seed"]} in registry')
    base = bases[-1]  # latest successful (re)run
    ckpt = os.path.join(base['run_dir'], f'params_{restore_epoch}.pkl')
    if not os.path.exists(ckpt):
        raise RuntimeError(f'base run {base["run_id"]} has no {ckpt}')
    registry.update(row['run_id'], depends_on=base['run_id'])
    return [a.replace(matrix.FROZEN_PLACEHOLDER, base['run_dir']) for a in args]


def run_dirs(save_dir, group, seed):
    return set(glob.glob(os.path.join(REPO_ROOT, save_dir, group, f'sd{int(seed):03d}_*')))


def check_outputs(run_dir, train_steps):
    """(ok, final_success, note) per spec Step 2 pass criteria."""
    problems = []
    train_csv = os.path.join(run_dir, 'train.csv')
    eval_csv = os.path.join(run_dir, 'eval.csv')
    final_success = ''
    if os.path.exists(train_csv):
        with open(train_csv) as f:
            for r in csv.DictReader(f):
                if any(_is_nan(v) for v in r.values()):
                    problems.append(f'NaN in train.csv at step {r.get("step")}')
                    break
    else:
        problems.append('no train.csv')
    if os.path.exists(eval_csv):
        with open(eval_csv) as f:
            rows = list(csv.DictReader(f))
        if rows and 'evaluation/overall_success' in rows[-1]:
            final_success = f'{100 * float(rows[-1]["evaluation/overall_success"]):.1f}'
    else:
        problems.append('no eval.csv')
    if not os.path.exists(os.path.join(run_dir, f'params_{train_steps}.pkl')):
        problems.append(f'no params_{train_steps}.pkl')
    return not problems, final_success, '; '.join(problems)


def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--log_file', default=None, help='Redirect main.py output here (default: inherit).')
    opts = parser.parse_args()

    row = registry.get(opts.run_id)
    args = shlex.split(row['command'])
    assert args[:2] == ['python', 'main.py'], row['command']
    args = args[2:]

    try:
        args = resolve_frozen_path(row, args)
    except RuntimeError as e:
        registry.update(row['run_id'], status='failed', finished_at=registry.now(), notes=str(e))
        print(f'[run_one] {e}', file=sys.stderr)
        return 1

    save_dir = flag_value(args, '--save_dir')
    group = flag_value(args, '--wandb_run_group')
    train_steps = matrix.train_steps_of(args)

    resolved = 'python main.py ' + ' '.join(shlex.quote(a) for a in args)
    registry.update(row['run_id'], status='running', started_at=registry.now(), finished_at='',
                    git_commit=git_commit(), machine=socket.gethostname(), job_id=job_id(),
                    command=resolved, notes='')

    env = dict(os.environ)
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    before = run_dirs(save_dir, group, row['seed'])
    log = open(opts.log_file, 'a') if opts.log_file else None
    proc = subprocess.Popen([sys.executable, 'main.py'] + args, cwd=REPO_ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT if log else None)

    def on_term(signum, _frame):
        proc.terminate()
        registry.update(row['run_id'], status='failed', finished_at=registry.now(),
                        notes=f'killed by signal {signum}')
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    run_dir = None
    while proc.poll() is None:
        if run_dir is None:
            new = run_dirs(save_dir, group, row['seed']) - before
            if len(new) == 1:
                run_dir = new.pop()
                registry.update(row['run_id'], run_dir=run_dir)
            elif len(new) > 1:
                print(f'[run_one] ambiguous new run dirs {new}', file=sys.stderr)
        time.sleep(5)
    if log:
        log.close()

    if run_dir is None:
        new = run_dirs(save_dir, group, row['seed']) - before
        run_dir = new.pop() if len(new) == 1 else ''
        registry.update(row['run_id'], run_dir=run_dir)

    if proc.returncode != 0 or not run_dir:
        registry.update(row['run_id'], status='failed', finished_at=registry.now(),
                        notes=f'main.py exit code {proc.returncode}' + ('' if run_dir else '; run_dir not found'))
        return 1
    ok, final_success, note = check_outputs(run_dir, train_steps)
    registry.update(row['run_id'], status='done' if ok else 'failed', finished_at=registry.now(),
                    final_success=final_success, notes=note)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
