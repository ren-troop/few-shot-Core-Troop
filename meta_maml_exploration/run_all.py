from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHODS = ("maml", "meta_sgd", "fixed_l2", "meta_l2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="依次运行四个可比实验并汇总结果")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--meta-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--download", action="store_true")
    return parser


def run_experiments(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent
    for method in METHODS:
        command = [
            sys.executable,
            str(project_dir / "train.py"),
            "--method",
            method,
            "--episodes",
            str(args.episodes),
            "--eval-episodes",
            str(args.eval_episodes),
            "--meta-batch",
            str(args.meta_batch),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--data-dir",
            args.data_dir,
            "--output-dir",
            str(Path(args.output_root) / method),
            "--n-way",
            str(args.n_way),
            "--k-shot",
            str(args.k_shot),
            "--q-query",
            str(args.q_query),
            "--inner-steps",
            str(args.inner_steps),
        ]
        if args.download:
            command.append("--download")
        print("运行：", " ".join(command))
        subprocess.run(command, check=True, cwd=project_dir)


def summarize(output_root: Path) -> None:
    summary_rows, learning_curves = [], []
    for method in METHODS:
        method_dir = output_root / method
        with (method_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary_rows.append(
            {
                "method": method,
                "test_accuracy_mean": summary["test_accuracy_mean"],
                "test_accuracy_ci95": summary["test_accuracy_ci95"],
                "test_loss_mean": summary["test_loss_mean"],
            }
        )
        learning_curves.append(pd.read_csv(method_dir / "training_log.csv"))

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_root / "comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for curve in learning_curves:
        method = str(curve["method"].iloc[0])
        smoothed = curve["train_query_accuracy"].rolling(20, min_periods=1).mean()
        axes[0].plot(curve["episode"], smoothed, label=method)
    axes[0].set_title("Training query accuracy")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        summary_frame["method"],
        summary_frame["test_accuracy_mean"],
        yerr=summary_frame["test_accuracy_ci95"],
        capsize=4,
    )
    axes[1].set_title("Held-out Omniglot tasks")
    axes[1].set_ylabel("Accuracy (95% CI)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    if not data_dir.is_absolute():
        data_dir = project_dir / data_dir
    if not output_root.is_absolute():
        output_root = project_dir / output_root
    args.data_dir = str(data_dir.resolve())
    args.output_root = str(output_root.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    run_experiments(args)
    summarize(output_root)
    print(f"全部完成，汇总文件位于：{output_root.resolve()}")


if __name__ == "__main__":
    main()
