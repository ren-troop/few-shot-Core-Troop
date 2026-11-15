"""Shared CLI definition for train.py and run_all.py."""

from __future__ import annotations

import argparse
import math

METHODS = ("maml", "meta_sgd", "meta_l2", "fixed2")
PROTOCOL = "omniglot_5way_1shot_background_eval"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    for name, default in {
        "seed": 42,
        "episodes": 600,
        "eval-episodes": 200,
        "meta-batch": 4,
        "n-way": 5,
        "k-shot": 1,
        "q-query": 5,
        "inner-steps": 1,
        "hidden-channels": 32,
        "log-every": 20,
        "val-every": 100,
        "val-episodes": 50,
        "split-seed": 2026,
        "num-threads": 4,
        "pretrain-epochs": 30,
        "pretrain-batch-size": 128,
        "fixed2-steps": 2,
    }.items():
        parser.add_argument("--" + name, type=int, default=default)

    for name, default in {
        "inner-lr": 0.4,
        "outer-lr": 1e-3,
        "initial-l2": 1e-3,
        "l2-outer-lr": 1e-2,
        "max-inner-lr": 1.0,
        "val-fraction": 0.2,
        "pretrain-lr": 1e-3,
        "pretrain-weight-decay": 5e-4,
        "fixed2-lr": 0.05,
    }.items():
        parser.add_argument("--" + name, type=float, default=default)

    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--fixed2-finetune-backbone", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")


def validate_args(args, parser: argparse.ArgumentParser, methods: list[str] | tuple[str, ...]) -> None:
    for key in (
        "episodes",
        "eval_episodes",
        "meta_batch",
        "n_way",
        "k_shot",
        "q_query",
        "inner_steps",
        "hidden_channels",
        "log_every",
        "val_episodes",
        "num_threads",
        "pretrain_epochs",
        "pretrain_batch_size",
        "fixed2_steps",
    ):
        if getattr(args, key) <= 0:
            parser.error(key.replace("_", "-") + " must be positive")

    for key in (
        "inner_lr",
        "outer_lr",
        "initial_l2",
        "l2_outer_lr",
        "max_inner_lr",
        "pretrain_lr",
        "pretrain_weight_decay",
        "fixed2_lr",
    ):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(key.replace("_", "-") + " must be finite and > 0")

    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be in (0, 1)")
    if args.val_every < 0:
        parser.error("val-every must be >= 0")
    if args.eval_episodes < 2 or args.val_episodes < 2:
        parser.error("eval-episodes and val-episodes must be at least 2 for a t interval")
    if args.k_shot + args.q_query > 20:
        parser.error("Omniglot has 20 images per character: k-shot + q-query <= 20")
    if args.first_order and "meta_l2" in methods:
        parser.error("Meta-L2 requires second-order gradients; omit --first-order")
