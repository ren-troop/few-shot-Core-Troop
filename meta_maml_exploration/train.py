from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from meta_experiment.data import OmniglotEpisodeSampler
from meta_experiment.learner import METHODS, MetaLearner
from meta_experiment.model import OmniglotConvNet
from meta_experiment.utils import choose_device, save_json, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Omniglot 上的 MAML 拓展实验")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--meta-batch", type=int, default=4)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--inner-lr", type=float, default=0.4)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--initial-l2", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--first-order", action="store_true")
    return parser


def evaluate(
    learner: MetaLearner,
    sampler: OmniglotEpisodeSampler,
    device: torch.device,
    episode_count: int,
) -> dict[str, float]:
    learner.eval()
    losses, accuracies = [], []
    for _ in range(episode_count):
        episode = sampler.sample().to(device)
        loss, accuracy = learner.episode_metrics(episode, training=False)
        losses.append(float(loss.detach().cpu()))
        accuracies.append(float(accuracy.detach().cpu()))
    accuracy_array = np.asarray(accuracies, dtype=np.float64)
    standard_error = accuracy_array.std(ddof=1) / math.sqrt(len(accuracy_array))
    return {
        "test_loss_mean": float(np.mean(losses)),
        "test_accuracy_mean": float(accuracy_array.mean()),
        "test_accuracy_std": float(accuracy_array.std(ddof=1)),
        "test_accuracy_ci95": float(1.96 * standard_error),
    }


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sampler = OmniglotEpisodeSampler(
        root=args.data_dir,
        background=True,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        download=args.download,
        seed=args.seed,
    )
    test_sampler = OmniglotEpisodeSampler(
        root=args.data_dir,
        background=False,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        download=args.download,
        seed=args.seed + 10000,
    )

    learner = MetaLearner(
        model=OmniglotConvNet(args.n_way, args.hidden_channels),
        method=args.method,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        initial_l2=args.initial_l2,
        first_order=args.first_order,
    ).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.outer_lr)

    history: list[dict[str, float | int | str]] = []
    learner.train()
    for episode_index in range(1, args.episodes + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_losses, batch_accuracies = [], []
        for _ in range(args.meta_batch):
            episode = train_sampler.sample().to(device)
            loss, accuracy = learner.episode_metrics(episode, training=True)
            batch_losses.append(loss)
            batch_accuracies.append(accuracy.detach())

        meta_loss = torch.stack(batch_losses).mean()
        meta_accuracy = torch.stack(batch_accuracies).mean()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=10.0)
        optimizer.step()

        row = {
            "episode": episode_index,
            "method": args.method,
            "train_query_loss": float(meta_loss.detach().cpu()),
            "train_query_accuracy": float(meta_accuracy.cpu()),
        }
        history.append(row)
        if episode_index == 1 or episode_index % args.log_every == 0:
            print(
                f"[{args.method}] episode={episode_index:04d} "
                f"loss={row['train_query_loss']:.4f} "
                f"acc={row['train_query_accuracy']:.4f}"
            )

    metrics = evaluate(learner, test_sampler, device, args.eval_episodes)
    configuration = vars(args).copy()
    configuration["device_used"] = str(device)
    summary = {"method": args.method, "configuration": configuration, **metrics}

    pd.DataFrame(history).to_csv(output_dir / "training_log.csv", index=False)
    hyperparameter_columns = ["method", "kind", "parameter", "mean", "std", "min", "max"]
    pd.DataFrame(
        learner.learned_hyperparameter_rows(), columns=hyperparameter_columns
    ).to_csv(output_dir / "learned_hyperparameters.csv", index=False)
    save_json(output_dir / "summary.json", summary)
    torch.save(
        {"state_dict": learner.state_dict(), "summary": summary},
        output_dir / "checkpoint.pt",
    )
    print(
        f"完成：test_acc={metrics['test_accuracy_mean']:.4f} "
        f"±{metrics['test_accuracy_ci95']:.4f} (95% CI)"
    )


if __name__ == "__main__":
    main()
