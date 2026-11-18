"""Download OGBench datasets (train + `-val`) into a dataset dir. Run on a node with internet access.

    python scripts/download_datasets.py --dataset_dir=~/datasets/ogbench
"""
import argparse
import os

import ogbench

DEFAULT = [f'antmaze-{m}-navigate-v0' for m in ('medium', 'large', 'giant', 'teleport')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--datasets', default=','.join(DEFAULT), help='Comma-separated names without `ogbench-`.')
    args = p.parse_args()
    # ogbench.download_datasets skips files that already exist.
    ogbench.download_datasets(args.datasets.split(','), dataset_dir=args.dataset_dir)
    d = os.path.expanduser(args.dataset_dir)
    for name in args.datasets.split(','):
        for suffix in ('', '-val'):
            path = os.path.join(d, f'{name}{suffix}.npz')
            print(f'{"ok" if os.path.exists(path) else "MISSING"} {path}')


if __name__ == '__main__':
    main()
