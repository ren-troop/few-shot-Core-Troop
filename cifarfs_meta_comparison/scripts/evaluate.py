#!/usr/bin/env python
"""Evaluate a trained checkpoint on the validation or test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metalab.config import load_config  # noqa: E402
from metalab.engine import Experiment  # noqa: E402
from metalab.utils import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tasks", type=int, default=600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.get("seed", 0)))
    experiment = Experiment(cfg, device=args.device)
    experiment.load_checkpoint(args.checkpoint, load_optimizer=False)
    metrics = experiment.evaluate(args.split, n_tasks=args.tasks, progress=True)
    metrics["checkpoint"] = Path(args.checkpoint).as_posix()
    metrics["split"] = args.split
    metrics["dataset"] = str(cfg.dataset.name)
    metrics["algorithm"] = str(cfg.algorithm.name)
    print(json.dumps(metrics, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
