"""Evaluate saved checkpoints of a finished run without retraining (spec 01 Step 6).

    python scripts/eval_checkpoint.py --run_dir=<run dir> --epochs=100000,200000 [--eval_episodes=50]
        [--complex_task_name=regions] [--seed=0] [--dataset_dir=...]

Rebuilds the agent from the run's `flags.json`, restores `params_<epoch>.pkl` with
`utils.flax_utils.restore_agent` (hierarchical checkpoints already contain the frozen modules, so
`load_agent_from_frozen` is not called), and runs the same evaluation as `main.py`:
`relabel_dataset` -> `infer_latent` -> `utils.evaluation.evaluate`, per task.

Writes one CSV per epoch next to the checkpoint:
    <run_dir>/eval_only_<epoch>[_<complex_task_name>]_sd<seed>.csv
with the same columns as `eval.csv`. Episode randomness differs from the training run, so results
match `eval.csv` only up to evaluation noise.
"""
import argparse
import csv
import importlib
import json
import os
import random
import sys
from collections import defaultdict

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
np.in1d = np.isin  # same monkey-patch as main.py
import ml_collections  # noqa: E402

from agents import agents  # noqa: E402
from utils.datasets import Dataset  # noqa: E402
from utils.env_utils import make_env_and_datasets, relabel_dataset  # noqa: E402
from utils.evaluation import evaluate  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402

# Agents that evaluate without a zero-shot latent (main.py, evaluation block).
NO_LATENT_AGENTS = ['hiql', 'iql', 'iqlbilinear', 'iqlintentions']
REGION_METRICS = ['total_discounted_return', 'total_return', 'goal_reached', 'in_goal_return', 'to_goal_return',
                  'in_goal_discounted_return', 'to_goal_discounted_return']


def load_run(run_dir, dataset_dir=None):
    """(saved flags, config, eval_env, zero-shot source dataset, dataset_class, example_batch) as in main.py."""
    with open(os.path.join(run_dir, 'flags.json')) as f:
        saved = json.load(f)
    config = ml_collections.ConfigDict(saved['agent'])
    dataset_dir = dataset_dir or saved.get('dataset_dir', '~/.ogbench/data')
    eval_env, train_dataset, val_dataset = make_env_and_datasets(
        saved['env_name'], frame_stack=config['frame_stack'], add_info=True, dataset_dir=dataset_dir)
    eval_env.unwrapped._add_noise_to_goal = False

    train_dataset = Dataset.create(**train_dataset)
    if val_dataset is not None:
        val_dataset = Dataset.create(**val_dataset)
        zero_shot_dataset_dict = val_dataset
    else:
        zero_shot_dataset_dict = train_dataset
    dataset_class = getattr(importlib.import_module('utils.datasets'), config['dataset_class'])
    train_dataset = dataset_class(train_dataset, config)
    return saved, config, eval_env, zero_shot_dataset_dict, dataset_class, train_dataset


def make_agent(config, train_dataset, seed):
    random.seed(seed)
    np.random.seed(seed)
    example_batch = train_dataset.sample(1)
    return agents[config['agent_name']].create(seed, example_batch, config)


def evaluate_all_tasks(agent, eval_env, env_name, config, zero_shot_dataset_dict, dataset_class,
                       eval_episodes, complex_task_name=None, eval_temperature=0.0, eval_gaussian=None):
    """Mirror of main.py's evaluation block (video disabled). Returns the eval.csv metrics dict."""
    num_zero_shot_samples = config.get('num_zero_shot_samples') or 100_000
    task_infos = (eval_env.unwrapped.task_infos if hasattr(eval_env.unwrapped, 'task_infos')
                  else eval_env.task_infos)
    metric_names = REGION_METRICS if complex_task_name == 'regions' else ['success']
    eval_metrics, overall = {}, defaultdict(list)
    for task_id in range(1, len(task_infos) + 1):
        if config['agent_name'] not in NO_LATENT_AGENTS:
            eval_env.reset(options=dict(task_id=task_id))
            zero_shot_dataset = relabel_dataset(env_name, eval_env, zero_shot_dataset_dict,
                                                complex_task_name=complex_task_name)
            zero_shot_dataset = dataset_class(Dataset.create(**zero_shot_dataset), config)
            assert zero_shot_dataset.size >= num_zero_shot_samples
            zero_shot_batch = zero_shot_dataset.sample(num_zero_shot_samples, idxs=np.arange(num_zero_shot_samples),
                                                       relabeling=False, augmentation=False)
            inferred_latent = np.asarray(agent.infer_latent(zero_shot_batch))
        else:
            inferred_latent = None
        eval_info, _, _ = evaluate(
            agent=agent, env=eval_env, task_id=task_id, inferred_latent=inferred_latent,
            num_eval_episodes=eval_episodes, num_video_episodes=0, eval_temperature=eval_temperature,
            eval_gaussian=eval_gaussian, complex_task_name=complex_task_name)
        for k, v in eval_info.items():
            if k in metric_names:
                eval_metrics[f'evaluation/{task_id}_{k}'] = v
                overall[k].append(v)
    for k, v in overall.items():
        eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
    return eval_metrics


def output_path(run_dir, epoch, complex_task_name, seed):
    suffix = f'_{complex_task_name}' if complex_task_name else ''
    return os.path.join(run_dir, f'eval_only_{epoch}{suffix}_sd{seed}.csv')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True)
    p.add_argument('--epochs', required=True, help='Comma-separated checkpoint steps.')
    p.add_argument('--complex_task_name', default=None, choices=[None, 'regions'])
    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset_dir', default=None, help='Default: the run\'s own --dataset_dir.')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    saved, config, eval_env, zs_dict, dataset_class, train_dataset = load_run(run_dir, args.dataset_dir)
    agent = make_agent(config, train_dataset, saved['seed'])
    for epoch in [int(e.replace('_', '')) for e in args.epochs.split(',')]:
        out = output_path(run_dir, epoch, args.complex_task_name, args.seed)
        if os.path.exists(out) and not args.overwrite:
            print(f'skip {out} (exists)')
            continue
        agent = restore_agent(agent, run_dir, epoch)
        np.random.seed(args.seed)
        random.seed(args.seed)
        metrics = evaluate_all_tasks(agent, eval_env, saved['env_name'], config, zs_dict, dataset_class,
                                     args.eval_episodes, args.complex_task_name,
                                     saved.get('eval_temperature', 0.0), saved.get('eval_gaussian'))
        metrics['step'] = epoch
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        overall = metrics.get('evaluation/overall_success')
        print(f'{run_dir} epoch {epoch}: ' + (f'overall_success={overall:.3f}' if overall is not None else str(metrics)),
              flush=True)


if __name__ == '__main__':
    main()
