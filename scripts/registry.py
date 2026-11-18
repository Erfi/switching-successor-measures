"""Run registry: one CSV row per run, shared by both machines through git.

All writes take a POSIX lock (fcntl.lockf, which works on NFS) on a sidecar lock file and replace
the CSV atomically, so concurrent SLURM tasks on the cluster can update their own rows safely.
"""
import contextlib
import csv
import fcntl
import os
import tempfile
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(REPO_ROOT, 'scripts', 'registry.csv')

COLUMNS = [
    'run_id', 'method', 'env', 'seed', 'stage', 'depends_on', 'command', 'run_dir', 'git_commit',
    'status', 'final_success', 'machine', 'job_id', 'started_at', 'finished_at', 'notes',
]
# planned -> queued (slurm) -> running -> done | failed
STATUSES = {'planned', 'queued', 'running', 'done', 'failed'}


def now():
    return datetime.now().isoformat(timespec='seconds')


@contextlib.contextmanager
def _locked(path=REGISTRY_PATH):
    with open(path + '.lock', 'a') as lock_file:
        fcntl.lockf(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in COLUMNS:
            row.setdefault(col, '')
    return rows


def _write(rows, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.registry.', suffix='.tmp')
    with os.fdopen(fd, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') for c in COLUMNS})
    os.replace(tmp, path)


def read_all(path=REGISTRY_PATH):
    with _locked(path):
        return _read(path)


def get(run_id, path=REGISTRY_PATH):
    for row in read_all(path):
        if row['run_id'] == run_id:
            return row
    raise KeyError(f'run_id {run_id!r} not in registry')


def add(row, path=REGISTRY_PATH):
    """Append a new row. Fails if run_id already exists."""
    unknown = set(row) - set(COLUMNS)
    assert not unknown, f'unknown columns {unknown}'
    with _locked(path):
        rows = _read(path)
        if any(r['run_id'] == row['run_id'] for r in rows):
            raise ValueError(f'run_id {row["run_id"]!r} already in registry')
        rows.append({c: str(row.get(c, '')) for c in COLUMNS})
        _write(rows, path)


def update(run_id, path=REGISTRY_PATH, **fields):
    unknown = set(fields) - set(COLUMNS)
    assert not unknown, f'unknown columns {unknown}'
    if 'status' in fields:
        assert fields['status'] in STATUSES, fields['status']
    with _locked(path):
        rows = _read(path)
        matches = [r for r in rows if r['run_id'] == run_id]
        if not matches:
            raise KeyError(f'run_id {run_id!r} not in registry')
        for k, v in fields.items():
            matches[0][k] = '' if v is None else str(v)
        _write(rows, path)
        return dict(matches[0])


def find(path=REGISTRY_PATH, **criteria):
    """Rows whose columns equal all criteria (compared as strings), in registry order."""
    return [r for r in read_all(path) if all(r[k] == str(v) for k, v in criteria.items())]


def next_run_id(base_id, path=REGISTRY_PATH):
    """`base_id` if free, else `base_id-r<n>` with the smallest free n (used for reruns)."""
    taken = {r['run_id'] for r in read_all(path)}
    if base_id not in taken:
        return base_id
    n = 1
    while f'{base_id}-r{n}' in taken:
        n += 1
    return f'{base_id}-r{n}'
