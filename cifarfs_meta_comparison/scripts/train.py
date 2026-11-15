#!/usr/bin/env python
"""Train one of the four methods from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metalab.config import load_config  # noqa: E402
from metalab.engine import Experiment  # noqa: E402
from metalab.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument("--output-dir", default=None, help="Override train.output_dir.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume from (for MAML/Meta-SGD/Meta-L2).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. --set train.meta_iters=100.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    seed_everything(int(cfg.get("seed", 0)))

    experiment = Experiment(cfg, device=args.device)
    if args.resume:
        if experiment.is_fixed2:
            raise SystemExit("Fixed-2 is pretrained from scratch; --resume is not used.")
        experiment.load_checkpoint(args.resume)

    experiment.train()
    print(f"Artifacts written to {Path(cfg.train.output_dir).resolve()}")


if __name__ == "__main__":
    main()
