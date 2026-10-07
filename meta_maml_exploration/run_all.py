from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

from meta_experiment.config import METHODS, add_common_arguments, validate_args


def build_parser():
    parser = argparse.ArgumentParser(description='Run comparable Omniglot methods')
    add_common_arguments(parser)
    parser.add_argument('--output-root', default='outputs_review_seed42')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--plots', action='store_true', help='Create local figures; do not commit them')
    return parser


def build_command(args, method):
    command = [sys.executable, str(Path(__file__).with_name('train.py')), '--method', method,
               '--output-dir', str(Path(args.output_root)/method)]
    for key, value in vars(args).items():
        if key in ('output_root', 'methods', 'plots'):
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def summarize(root, methods):
    import pandas as pd
    rows = []
    for method in methods:
        s = json.loads((root/method/'summary.json').read_text(encoding='utf-8'))
        rows.append(dict(method=method, test_accuracy_mean=s['test_accuracy_mean'],
                         test_accuracy_ci95=s['test_accuracy_ci95'], test_loss_mean=s['test_loss_mean']))
    pd.DataFrame(rows).to_csv(root/'comparison.csv', index=False)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser, args.methods)
    if len(args.methods) != len(set(args.methods)):
        parser.error('Duplicate methods')
    project = Path(__file__).resolve().parent
    # All relative paths are based on this project, also when launched from PyCharm.
    args.data_dir = str((project/args.data_dir).resolve())
    args.output_root = str((project/args.output_root).resolve())
    root = Path(args.output_root)
    if root.exists() and any(root.iterdir()):
        parser.error('output-root is not empty; choose a new name')
    root.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        command = build_command(args, method)
        print('Running:', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True, cwd=project)
    summarize(root, args.methods)
    if args.plots:
        from plot_results import plot_results
        plot_results(root)
    print(f'All methods completed: {root}', flush=True)


if __name__ == '__main__':
    main()
