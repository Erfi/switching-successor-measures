"""Run matrix for spec 01 (replication). Single source of truth for which runs exist and their commands.

Per-(method, env) arguments are parsed from `hyperparameters.sh` so commands match the published
script verbatim; only the evaluation/logging flags of spec 01 Step 4 are appended.
"""
import os
import shlex

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYPERPARAMETERS_SH = os.path.join(REPO_ROOT, 'hyperparameters.sh')

FROZEN_PLACEHOLDER = '{FROZEN_PATH}'
_PATH_PLACEHOLDERS = ('PATH_TO_PRETRAINED_FB_MODEL', 'PATH_TO_PRETRAINED_FB_pi_SWITCH_REPRESENTATIONS')

WANDB_PROJECT = 'HFB'
DATASET_DIRS = {
    'workstation': '~/.ogbench/data',
    'cluster': '~/datasets/ogbench',
}
BACKEND_MACHINE = {'local': 'workstation', 'slurm': 'cluster'}

ENVS = {
    'medium': 'ogbench-antmaze-medium-navigate-v0',
    'large': 'ogbench-antmaze-large-navigate-v0',
    'giant': 'ogbench-antmaze-giant-navigate-v0',
    'teleport': 'ogbench-antmaze-teleport-navigate-v0',
}

# method -> agent file and, for hierarchical methods, the base method they freeze.
METHODS = {
    'fb': dict(agent='agents/fb.py', base=None),
    'fbpis_nonh': dict(agent='agents/fbpiswitch_nonhierarchical.py', base=None),
    'hiql': dict(agent='agents/hiql.py', base=None),
    'icvf': dict(agent='agents/icvf.py', base=None),
    'onestep_fb': dict(agent='agents/onestep_fb.py', base=None),
    'fb_h': dict(agent='agents/fb_hierarchical.py', base='fb'),
    'fbpis': dict(agent='agents/fbpiswitch.py', base='fbpis_nonh'),
}

FB_FAMILY = ['fb', 'fbpis_nonh', 'fb_h', 'fbpis']
ALL5 = [0, 1, 2, 3, 4]

# stage -> list of (methods, envs, seeds) blocks (spec 01 Step 5; smoke = Step 2).
STAGES = {
    'smoke': [(['fb', 'fbpis_nonh', 'hiql', 'icvf', 'onestep_fb', 'fb_h', 'fbpis'], ['medium'], [0])],
    'smoke_cluster': [(['fb', 'fb_h'], ['medium'], [0])],  # separate run_ids from the workstation smoke
    'A': [(['fb', 'fbpis_nonh', 'hiql', 'fb_h', 'fbpis'], ['medium'], [0])],
    'B1': [(FB_FAMILY, ['medium', 'large'], [0, 1, 2])],
    'B2': [(FB_FAMILY, ['medium', 'large'], [3, 4]), (['hiql'], ['medium', 'large'], ALL5)],
    'C': [(FB_FAMILY + ['hiql'], ['teleport'], ALL5)],
    'C_opt': [(['icvf', 'onestep_fb'], ['medium', 'large'], ALL5)],
    'D': [(FB_FAMILY, ['giant'], ALL5)],
}

SMOKE_FLAGS = ['--train_steps=2000', '--log_interval=500', '--eval_interval=2000', '--save_interval=2000',
               '--eval_episodes=1', '--video_episodes=0', '--enable_wandb=0']
SMOKE_RESTORE_EPOCH = 2000
DEFAULT_TRAIN_STEPS = 1_000_000
DEFAULT_RESTORE_EPOCH = 1_000_000


def parse_hyperparameters(path=HYPERPARAMETERS_SH):
    """{(agent_file, env_name): [args after `python main.py`]} from hyperparameters.sh."""
    table = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('python main.py'):
                continue
            args = shlex.split(line)[2:]
            env = _flag_value(args, '--env_name')
            agent = _flag_value(args, '--agent')
            args = [_replace_placeholders(a) for a in args]
            key = (agent, env)
            assert key not in table, f'duplicate line in hyperparameters.sh for {key}'
            table[key] = args
    return table


def _replace_placeholders(arg):
    for p in _PATH_PLACEHOLDERS:
        arg = arg.replace(p, FROZEN_PLACEHOLDER)
    return arg


def _flag_value(args, name):
    """Value of the last occurrence of `--name=value` (absl: last one wins), or None."""
    for a in reversed(args):
        if a.startswith(name + '='):
            return a.split('=', 1)[1]
    return None


def train_steps_of(args):
    v = _flag_value(args, '--train_steps')
    return int(v.replace('_', '')) if v is not None else DEFAULT_TRAIN_STEPS


def is_smoke(stage):
    return stage.startswith('smoke')


def group_name(method, env, stage):
    return f'smoke_{method}_{env}' if is_smoke(stage) else f'{method}_{env}'


def save_dir_of(stage):
    return 'exp_logs_smoke' if is_smoke(stage) else 'exp_logs'


def build_args(method, env, seed, stage, machine, table=None):
    """Full argument list for main.py (without the interpreter)."""
    table = table if table is not None else parse_hyperparameters()
    key = (METHODS[method]['agent'], ENVS[env])
    if key not in table:
        raise KeyError(f'no hyperparameters.sh line for {key}')
    args = list(table[key])
    common = [
        f'--seed={seed}',
        f'--dataset_dir={DATASET_DIRS[machine]}',
        f'--save_dir={save_dir_of(stage)}',
        f'--wandb_run_group={group_name(method, env, stage)}',
    ]
    if is_smoke(stage):
        extra = SMOKE_FLAGS[:]
        if METHODS[method]['base'] is not None:
            extra.append(f'--agent.restore_epoch={SMOKE_RESTORE_EPOCH}')
    else:
        extra = [
            '--eval_episodes=50', '--video_episodes=0', f'--eval_interval={train_steps_of(args)}',
            '--enable_wandb=1', f'--wandb_project={WANDB_PROJECT}',
        ]
    return args + common + extra


def build_command(method, env, seed, stage, machine, table=None):
    return 'python main.py ' + ' '.join(shlex.quote(a) for a in build_args(method, env, seed, stage, machine, table))


def stage_runs(stage, methods=None, envs=None, seeds=None):
    """[(method, env, seed)] for a stage, base methods before hierarchical ones, optionally filtered."""
    runs = []
    for block_methods, block_envs, block_seeds in STAGES[stage]:
        for m in block_methods:
            for e in block_envs:
                for s in block_seeds:
                    if methods and m not in methods:
                        continue
                    if envs and e not in envs:
                        continue
                    if seeds is not None and s not in seeds:
                        continue
                    runs.append((m, e, s))
    runs.sort(key=lambda r: METHODS[r[0]]['base'] is not None)  # stable: bases first
    return runs


def base_run_id(stage, method, env, seed):
    return f'{stage}-{method}-{env}-sd{seed}'


if __name__ == '__main__':
    # Print every command of every stage for eyeballing against hyperparameters.sh.
    t = parse_hyperparameters()
    for st in STAGES:
        for m, e, s in stage_runs(st):
            print(st, build_command(m, e, s, st, 'cluster', t))
