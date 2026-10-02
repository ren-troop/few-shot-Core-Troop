from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
import torch
import torchvision

from meta_experiment.config import METHODS, PROTOCOL, add_common_arguments, validate_args
from meta_experiment.data import OmniglotEpisodeSampler
from meta_experiment.learner import MetaLearner
from meta_experiment.model import OmniglotConvNet
from meta_experiment.utils import choose_device, save_json, seed_everything

PROJECT = Path(__file__).resolve().parent
HP_COLUMNS = ['method', 'kind', 'scope', 'parameter', 'count', 'mean', 'std', 'min', 'max',
              'negative_count', 'zero_count', 'positive_count', 'negative_fraction']


def build_parser():
    parser = argparse.ArgumentParser(description='Omniglot meta-learning experiments')
    add_common_arguments(parser)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output-dir', required=True)
    return parser


def build_optimizer(learner, args):
    raw_ids = {id(p) for p in learner.raw_l2.parameters()}
    groups = [dict(params=[p for p in learner.parameters() if id(p) not in raw_ids],
                   lr=args.outer_lr, name='network_and_step_sizes')]
    if raw_ids:
        groups.append(dict(params=list(learner.raw_l2.parameters()), lr=args.l2_outer_lr,
                           eps=args.l2_adam_eps, name='raw_l2'))
    return torch.optim.Adam(groups, lr=args.outer_lr)


def evaluate(learner, sampler, device, episode_count, prefix='test'):
    was_training = learner.training
    rng_state = sampler.rng.getstate()
    learner.eval()
    rows = []
    try:
        # Do not use no_grad(): support-set adaptation requires autograd.
        for i in range(episode_count):
            loss, accuracy = learner.episode_metrics(sampler.sample().to(device), training=False)
            rows.append(dict(task=i, loss=float(loss.detach().cpu()),
                             accuracy=float(accuracy.detach().cpu())))
    finally:
        learner.train(was_training)
        sampler.rng.setstate(rng_state)  # Same validation tasks at every checkpoint.
    values = np.array([x['accuracy'] for x in rows], dtype=float)
    if not np.isfinite(values).all() or not all(math.isfinite(x['loss']) for x in rows):
        raise RuntimeError('Non-finite evaluation metrics')
    sd = values.std(ddof=1)
    metrics = {f'{prefix}_loss_mean': float(np.mean([x['loss'] for x in rows])),
               f'{prefix}_accuracy_mean': float(values.mean()),
               f'{prefix}_accuracy_std': float(sd),
               f'{prefix}_accuracy_ci95': float(t.ppf(.975, len(values)-1)*sd/math.sqrt(len(values)))}
    return metrics, rows


def portable_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        # No username or machine-specific parent directory is written to summaries.
        return '<external>/' + path.name


def l2_values(learner):
    return {name.replace('__', '.'): (raw.detach().item(), torch.nn.functional.softplus(raw).detach().item())
            for name, raw in learner.raw_l2.items()}


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser, [args.method])
    seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    # Prevent old weights/results being silently mixed with a new run.
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error('output-dir is not empty. Choose a fresh directory for each run.')
    output_dir.mkdir(parents=True, exist_ok=True)
    common = dict(root=args.data_dir, n_way=args.n_way, k_shot=args.k_shot,
                  q_query=args.q_query, download=args.download,
                  val_fraction=args.val_fraction, split_seed=args.split_seed)
    train_sampler = OmniglotEpisodeSampler(**common, background=True, partition='train', seed=args.seed)
    val_sampler = OmniglotEpisodeSampler(**common, background=True, partition='validation', seed=args.seed+20000)
    test_sampler = OmniglotEpisodeSampler(**common, background=False, seed=args.seed+10000)
    if set(train_sampler.classes) & set(val_sampler.classes):
        raise RuntimeError('Training/validation character overlap')
    learner = MetaLearner(OmniglotConvNet(args.n_way, args.hidden_channels), args.method,
                          args.inner_lr, args.inner_steps, args.initial_l2, args.first_order,
                          args.meta_sgd_positive).to(device)
    optimizer = build_optimizer(learner, args)
    config = vars(args).copy()
    config.update(data_dir=portable_path(args.data_dir), output_dir=portable_path(output_dir),
                  device_used=str(device))
    split = dict(train_class_ids=train_sampler.classes, validation_class_ids=val_sampler.classes,
                 test_source='official_evaluation', test_class_count=len(test_sampler.classes))
    save_json(output_dir/'split.json', split)
    save_json(output_dir/'configuration.json', dict(protocol_version=PROTOCOL, configuration=config))
    signature = hashlib.sha256(str(split['train_class_ids']).encode()).hexdigest()
    trajectory = []
    for name, (raw, coefficient) in l2_values(learner).items():
        trajectory.append(dict(episode=0, parameter=name, raw_l2=raw, coefficient=coefficient,
                               relative_change=0., raw_grad_before_clip=None, raw_step=None))
    history, validation = [], []
    start = time.perf_counter()
    learner.train()
    with (output_dir/'training_log.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = None
        for step in range(1, args.episodes+1):
            optimizer.zero_grad(set_to_none=True)
            losses, accuracies = [], []
            for _ in range(args.meta_batch):
                loss, acc = learner.episode_metrics(train_sampler.sample().to(device), training=True)
                losses.append(loss)
                accuracies.append(acc.detach())
            meta_loss = torch.stack(losses).mean()
            meta_loss.backward()
            before = l2_values(learner)
            gradients = {}
            for key, raw in learner.raw_l2.items():
                if raw.grad is None:
                    raise RuntimeError(f'Meta-L2 gradient disconnected: {key}')
                gradients[key.replace('__', '.')] = raw.grad.detach().item()
            if not torch.isfinite(meta_loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                                    for p in learner.parameters()):
                raise RuntimeError(f'Non-finite loss/gradient at episode {step}')
            raw_norm = math.sqrt(sum(v*v for v in gradients.values())) if gradients else None
            total_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=10.)
            optimizer.step()
            row = dict(episode=step, method=args.method, train_query_loss=meta_loss.detach().item(),
                       train_query_accuracy=torch.stack(accuracies).mean().item(),
                       raw_l2_grad_norm=raw_norm, all_grad_norm_before_clip=float(total_norm))
            for name, (raw, coefficient) in l2_values(learner).items():
                row[f'l2::{name}'] = coefficient
                row[f'raw_l2_grad::{name}'] = gradients[name]
                trajectory.append(dict(episode=step, parameter=name, raw_l2=raw, coefficient=coefficient,
                                       relative_change=coefficient/args.initial_l2-1.,
                                       raw_grad_before_clip=gradients[name], raw_step=raw-before[name][0]))
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            history.append(row)
            if step == 1 or step % args.log_every == 0:
                diagnostic = f' raw_l2_grad_norm={raw_norm:.3e}' if raw_norm is not None else ''
                print(f'[{args.method}] episode={step:04d} loss={row["train_query_loss"]:.4f}'
                      f' acc={row["train_query_accuracy"]:.4f}{diagnostic}', flush=True)
                if before:
                    print('  L2: ' + ', '.join(f'{k}={v[1]:.6e}' for k,v in l2_values(learner).items()), flush=True)
            if args.val_every and (step % args.val_every == 0 or step == args.episodes):
                metrics, _ = evaluate(learner, val_sampler, device, args.val_episodes, 'val')
                validation.append(dict(episode=step, **metrics))
                pd.DataFrame(validation).to_csv(output_dir/'validation_log.csv', index=False)
                print(f'  validation: {metrics["val_accuracy_mean"]:.4f}', flush=True)
            if trajectory and (step % args.log_every == 0 or step == args.episodes):
                pd.DataFrame(trajectory).to_csv(output_dir/'l2_trajectory.csv', index=False)
    metrics, tasks = evaluate(learner, test_sampler, device, args.eval_episodes)
    pd.DataFrame(tasks).to_csv(output_dir/'test_tasks.csv', index=False)
    hp = pd.DataFrame(learner.learned_hyperparameter_rows(), columns=HP_COLUMNS)
    hp.to_csv(output_dir/'learned_hyperparameters.csv', index=False)
    if args.method == 'meta_sgd':
        hp[['parameter', 'count', 'negative_count', 'zero_count', 'positive_count',
            'negative_fraction']].to_csv(output_dir/'learning_rate_signs.csv', index=False)
    summary = dict(method=args.method, protocol_version=PROTOCOL, configuration=config,
                   split_signature=signature, selection_policy='final_episode_no_test_selection',
                   ci_definition='Student t interval across sampled test tasks for one trained model',
                   elapsed_seconds=time.perf_counter()-start,
                   environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                    torchvision=torchvision.__version__, numpy=np.__version__,
                                    pandas=pd.__version__, platform=platform.system(),
                                    device_name=torch.cuda.get_device_name() if device.type=='cuda' else 'CPU'),
                   l2_diagnostics=dict(gradient_connected=bool(learner.raw_l2),
                                       nonzero_gradient_episodes=sum(bool(x['raw_l2_grad_norm']) for x in history),
                                       initial_l2=args.initial_l2, final_values=l2_values(learner)), **metrics)
    save_json(output_dir/'summary.json', summary)
    if args.save_checkpoint:
        torch.save(dict(state_dict=learner.state_dict(), summary=summary), output_dir/'checkpoint.pt')
    print(f'Finished: test_acc={metrics["test_accuracy_mean"]:.4f} +/- {metrics["test_accuracy_ci95"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
