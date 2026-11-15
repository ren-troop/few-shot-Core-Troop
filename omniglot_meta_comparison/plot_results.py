from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(root: Path) -> None:
    root = Path(root)
    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for file in sorted(root.glob("*/summary.json")):
        summary = json.loads(file.read_text(encoding="utf-8"))
        summaries.append(summary)
        validation = file.parent / "validation_log.csv"
        if validation.exists():
            frame = pd.read_csv(validation)
            axes[0].plot(
                frame["episode"],
                frame["val_accuracy_mean"],
                marker="o",
                label=summary["method"],
            )
    axes[0].set(
        title="Held-out background validation",
        xlabel="Episode",
        ylabel="Accuracy",
    )
    if axes[0].lines:
        axes[0].legend()
    axes[1].bar(
        [summary["method"] for summary in summaries],
        [summary["test_accuracy_mean"] for summary in summaries],
        yerr=[summary["test_accuracy_ci95"] for summary in summaries],
        capsize=3,
    )
    axes[1].set(
        title="Final test accuracy (task-level 95% CI)",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sgd = root / "meta_sgd" / "training_log.csv"
    if sgd.exists():
        frame = pd.read_csv(sgd)
        axes[0].plot(frame["episode"], frame["mean_inner_lr"], color="#ff7f0e")
        axes[0].set(title="Meta-SGD mean inner LR", xlabel="Episode", ylabel="LR")
    l2 = root / "meta_l2" / "training_log.csv"
    if l2.exists():
        frame = pd.read_csv(l2)
        axes[1].plot(frame["episode"], frame["l2_lambda"], color="#2ca02c")
        axes[1].set(title="Meta-L2 learned lambda", xlabel="Episode", ylabel="lambda")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "hyperparameters.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    plot_results(parser.parse_args().root)
