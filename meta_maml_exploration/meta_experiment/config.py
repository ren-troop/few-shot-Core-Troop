"""One shared definition prevents run_all.py from dropping training arguments."""
import argparse
import math

METHODS = ("maml", "meta_sgd", "fixed_l2", "meta_l2")
PROTOCOL = "review_v2_background_class_holdout"


def add_common_arguments(parser):
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for name, default in {
        "seed": 42, "episodes": 600, "eval-episodes": 200, "meta-batch": 4,
        "n-way": 5, "k-shot": 1, "q-query": 5, "inner-steps": 1,
        "hidden-channels": 32, "log-every": 20, "val-every": 100,
        "val-episodes": 50, "split-seed": 2026, "num-threads": 4,
    }.items():
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in {
        "inner-lr": .4, "outer-lr": 1e-3, "initial-l2": 1e-4,
        "l2-outer-lr": 1e-2, "l2-adam-eps": 1e-8, "val-fraction": .2,
    }.items():
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--meta-sgd-positive", action="store_true",
                        help="Use softplus-constrained positive inner steps (a separate variant).")
    parser.add_argument("--save-checkpoint", action="store_true")


def validate_args(args, parser, methods):
    for key in ("episodes", "eval_episodes", "meta_batch", "n_way", "k_shot", "q_query",
                "inner_steps", "hidden_channels", "log_every", "val_episodes", "num_threads"):
        if getattr(args, key) <= 0:
            parser.error(key.replace("_", "-") + " must be positive")
    for key in ("inner_lr", "outer_lr", "initial_l2", "l2_outer_lr", "l2_adam_eps"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(key.replace("_", "-") + " must be finite and > 0")
    if args.eval_episodes < 2 or args.val_episodes < 2:
        parser.error("At least 2 evaluation/validation tasks are required for a CI")
    if args.val_every < 0 or not 0 < args.val_fraction < 1:
        parser.error("val-every must be >= 0 and val-fraction must be in (0, 1)")
    if args.k_shot + args.q_query > 20:
        parser.error("Omniglot has 20 images per character: k-shot + q-query must be <= 20")
    if args.first_order and "meta_l2" in methods:
        parser.error("Meta-L2 requires second-order gradients. Omit --first-order, or select other --methods.")
