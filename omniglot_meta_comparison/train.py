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
from meta_experiment.learner import Fixed2Learner, MetaLearner
from meta_experiment.model import OmniglotConvNet
from meta_experiment.utils import choose_device, save_json, seed_everything

PROJECT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Omniglot four-way meta-learning comparison")
    add_common_arguments(parser)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def portable_path(value: str) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT).as_posix()
    except ValueError:
        return "<external>/" + path.name


def evaluate(learner, sampler, device, episode_count: int, prefix: str = "test"):
    was_training = learner.training
    rng_state = sampler.rng.getstate()
    learner.eval()
    rows = []
    try:
        # Adaptation needs autograd even during evaluation.
        for index in range(episode_count):
            loss, accuracy = learner.episode_metrics(
                sampler.sample().to(device), training=False
            )
            rows.append(
                {
                    "task": index,
                    "loss": float(loss.detach().cpu()),
                    "accuracy": float(accuracy.detach().cpu()),
                }
            )
    finally:
        learner.train(was_training)
        sampler.rng.setstate(rng_state)

    values = np.array([row["accuracy"] for row in rows], dtype=float)
    if not np.isfinite(values).all() or not all(math.isfinite(row["loss"]) for row in rows):
        raise RuntimeError("Non-finite evaluation metric")
    sd = float(values.std(ddof=1))
    ci95 = float(t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
    metrics = {
        f"{prefix}_loss_mean": float(np.mean([row["loss"] for row in rows])),
        f"{prefix}_accuracy_mean": float(values.mean()),
        f"{prefix}_accuracy_std": sd,
        f"{prefix}_accuracy_ci95": ci95,
    }
    return metrics, rows


def build_optimizer(learner: MetaLearner, args):
    lambda_param = learner.l2_parameter()
    base = [p for p in learner.parameters() if p is not lambda_param]
    groups = [{"params": base, "lr": args.outer_lr, "name": "network"}]
    if lambda_param is not None:
        groups.append(
            {"params": [lambda_param], "lr": args.l2_outer_lr, "name": "l2_lambda"}
        )
    return torch.optim.Adam(groups)


def build_samplers(args, device: torch.device):
    common = dict(
        root=args.data_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        download=args.download,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    train = OmniglotEpisodeSampler(**common, background=True, partition="train", seed=args.seed)
    val = OmniglotEpisodeSampler(
        **common, background=True, partition="validation", seed=args.seed + 20_000
    )
    test = OmniglotEpisodeSampler(
        **common, background=False, partition="all", seed=args.seed + 10_000
    )
    if set(train.classes) & set(val.classes):
        raise RuntimeError("Training and validation characters overlap")
    return train, val, test


def write_summary(
    output_dir: Path,
    args,
    learner,
    split: dict,
    test_metrics: dict,
    elapsed: float,
    device: torch.device,
) -> None:
    summary = {
        "method": args.method,
        "protocol_version": PROTOCOL,
        "configuration": {
            **vars(args),
            "data_dir": portable_path(args.data_dir),
            "output_dir": portable_path(output_dir),
            "device_used": str(device),
        },
        "split_signature": hashlib.sha256(str(split["train_class_ids"]).encode()).hexdigest(),
        "selection_policy": "final_episode_no_test_selection",
        "ci_definition": "Student t interval across sampled test tasks for one trained model",
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.system(),
            "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        },
        **learner.hyperparameter_summary(),
        **test_metrics,
    }
    save_json(output_dir / "summary.json", summary)


def train_meta(args, device, train_sampler, val_sampler, test_sampler, output_dir):
    model = OmniglotConvNet(args.n_way, args.hidden_channels).to(device)
    learner = MetaLearner(
        model,
        args.method,
        args.inner_lr,
        args.inner_steps,
        args.initial_l2,
        args.first_order,
        args.max_inner_lr,
    ).to(device)
    optimizer = build_optimizer(learner, args)
    start = time.perf_counter()
    learner.train()
    validation = []
    training_log = output_dir / "training_log.csv"
    with training_log.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for episode in range(1, args.episodes + 1):
            optimizer.zero_grad(set_to_none=True)
            losses, accuracies = [], []
            for _ in range(args.meta_batch):
                loss, accuracy = learner.episode_metrics(
                    train_sampler.sample().to(device), training=True
                )
                losses.append(loss)
                accuracies.append(accuracy.detach())
            meta_loss = torch.stack(losses).mean()
            meta_loss.backward()
            if not torch.isfinite(meta_loss):
                raise RuntimeError(f"Non-finite loss at episode {episode}")
            for parameter in learner.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite gradient at episode {episode}")
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=10.0)
            optimizer.step()

            hyper = learner.hyperparameter_summary()
            row = {
                "episode": episode,
                "train_query_loss": float(meta_loss.detach().cpu()),
                "train_query_accuracy": float(torch.stack(accuracies).mean().cpu()),
                "mean_inner_lr": hyper.get("mean_inner_lr", ""),
                "l2_lambda": hyper.get("l2_lambda", ""),
            }
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

            if episode == 1 or episode % args.log_every == 0:
                extra = ""
                if "mean_inner_lr" in hyper:
                    extra = f" mean_lr={hyper['mean_inner_lr']:.4f}"
                if "l2_lambda" in hyper:
                    extra += f" lambda={hyper['l2_lambda']:.3e}"
                print(
                    f"[{args.method}] episode={episode:04d} "
                    f"loss={row['train_query_loss']:.4f} "
                    f"acc={row['train_query_accuracy']:.4f}{extra}",
                    flush=True,
                )

            if args.val_every and (episode % args.val_every == 0 or episode == args.episodes):
                metrics, _ = evaluate(learner, val_sampler, device, args.val_episodes, "val")
                validation.append({"episode": episode, **metrics})
                pd.DataFrame(validation).to_csv(output_dir / "validation_log.csv", index=False)
                print(f"  validation: {metrics['val_accuracy_mean']:.4f}", flush=True)

    test_metrics, test_rows = evaluate(
        learner, test_sampler, device, args.eval_episodes, "test"
    )
    pd.DataFrame(test_rows).to_csv(output_dir / "test_tasks.csv", index=False)
    elapsed = time.perf_counter() - start
    split = {
        "train_class_ids": train_sampler.classes,
        "validation_class_ids": val_sampler.classes,
        "test_source": "official_evaluation",
        "test_class_count": len(test_sampler.classes),
    }
    save_json(output_dir / "split.json", split)
    write_summary(output_dir, args, learner, split, test_metrics, elapsed, device)
    if args.save_checkpoint:
        torch.save({"state_dict": learner.state_dict(), "args": vars(args)}, output_dir / "checkpoint.pt")
    return test_metrics


def train_fixed2(args, device, train_sampler, val_sampler, test_sampler, output_dir):
    model = OmniglotConvNet(len(train_sampler.classes), args.hidden_channels).to(device)
    learner = Fixed2Learner(
        model,
        n_way=args.n_way,
        inner_steps=args.fixed2_steps,
        inner_lr=args.fixed2_lr,
        freeze_backbone=not args.fixed2_finetune_backbone,
    ).to(device)
    start = time.perf_counter()
    learner.pretrain(
        train_sampler,
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        weight_decay=args.pretrain_weight_decay,
        device=device,
        log_path=output_dir / "training_log.csv",
    )
    if args.save_checkpoint:
        torch.save({"state_dict": learner.state_dict(), "args": vars(args)}, output_dir / "pretrain.pt")

    metrics, _ = evaluate(learner, val_sampler, device, args.val_episodes, "val")
    pd.DataFrame([{"episode": args.pretrain_epochs, **metrics}]).to_csv(
        output_dir / "validation_log.csv", index=False
    )
    print(f"  validation: {metrics['val_accuracy_mean']:.4f}", flush=True)

    test_metrics, test_rows = evaluate(
        learner, test_sampler, device, args.eval_episodes, "test"
    )
    pd.DataFrame(test_rows).to_csv(output_dir / "test_tasks.csv", index=False)
    elapsed = time.perf_counter() - start
    split = {
        "train_class_ids": train_sampler.classes,
        "validation_class_ids": val_sampler.classes,
        "test_source": "official_evaluation",
        "test_class_count": len(test_sampler.classes),
    }
    save_json(output_dir / "split.json", split)
    write_summary(output_dir, args, learner, split, test_metrics, elapsed, device)
    return test_metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser, [args.method])
    seed_everything(args.seed)
    torch.set_num_threads(args.num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = choose_device(args.device)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output-dir must be empty; choose a fresh directory for each run")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sampler, val_sampler, test_sampler = build_samplers(args, device)
    print(
        f"Omniglot: train={len(train_sampler.classes)} "
        f"val={len(val_sampler.classes)} test={len(test_sampler.classes)} classes",
        flush=True,
    )
    if args.method == "fixed2":
        metrics = train_fixed2(args, device, train_sampler, val_sampler, test_sampler, output_dir)
    else:
        metrics = train_meta(args, device, train_sampler, val_sampler, test_sampler, output_dir)
    print(
        f"Finished: test_acc={metrics['test_accuracy_mean']:.4f} "
        f"+/- {metrics['test_accuracy_ci95']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
