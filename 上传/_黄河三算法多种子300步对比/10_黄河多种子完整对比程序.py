from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from full_comparison_experiments import (
    METHOD_COLORS,
    METHOD_MARKERS,
    UPDATE_LABELS,
    run_one_config,
    selected_methods,
    setup_plot_style,
)
from public_region_maml_meta_sgd import make_region_dataset, parse_list


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed Yellow River full comparison: second-order MAML, "
            "FO-MAML, Meta-SGD, with and without semi-supervised inner loop."
        )
    )
    parser.add_argument("--dataset", choices=["california", "csv"], default="csv")
    parser.add_argument("--data-dir", default="sklearn_data")
    parser.add_argument(
        "--csv-path",
        default="yellow_river_data/yellow_river_hydromlyr_model_ready.csv",
    )
    parser.add_argument("--region-column", default="region_id")
    parser.add_argument("--target-column", default="q")
    parser.add_argument("--feature-columns", default="")
    parser.add_argument("--drop-columns", default="date,is_natural")
    parser.add_argument("--region-bins", type=int, default=4)
    parser.add_argument("--min-samples-per-region", type=int, default=40)
    parser.add_argument("--eval-region-fraction", type=float, default=0.25)

    parser.add_argument(
        "--methods",
        default="Second-order MAML,FO-MAML,Meta-SGD",
        help="Comma-separated method labels.",
    )
    parser.add_argument(
        "--inner-updates",
        default="supervised,semi-supervised",
        help="Comma-separated inner update modes.",
    )
    parser.add_argument("--inner-steps-list", type=parse_int_list, default=parse_int_list("3"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("28,42,2026"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--unlabeled-per-task", type=int, default=16)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--min-meta-lr", type=float, default=1e-6)
    parser.add_argument("--clamp-meta-lr", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="yellow_river_multiseed_full_outputs")
    parser.add_argument("--results-csv", default="yellow_river_multiseed_full_results.csv")
    parser.add_argument("--summary-csv", default="yellow_river_multiseed_full_summary.csv")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "seed",
        "method",
        "algorithm",
        "create_graph",
        "first_order",
        "inner_update",
        "inner_update_label",
        "inner_steps",
        "step",
        "dataset",
        "train_regions",
        "eval_regions",
        "feature_count",
        "target",
        "meta_batch",
        "shots",
        "queries",
        "unlabeled_per_task",
        "ssl_weight_used",
        "outer_lr",
        "inner_lr",
        "support_loss",
        "ssl_loss",
        "inner_loss",
        "meta_loss",
        "eval_mse_norm",
        "eval_rmse_norm",
        "eval_mse_raw",
        "eval_rmse_raw",
        "eval_mae_raw",
        "best_eval_mse_norm",
        "grad_norm",
        "elapsed_sec",
        "avg_step_time_sec",
        "cuda_peak_memory_mb",
        "meta_lr_mean",
        "meta_lr_min",
        "meta_lr_max",
        "meta_lr_nonpositive_count",
    ]
    fieldnames = []
    seen = set()
    for name in preferred:
        if name not in seen:
            fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results_path: Path, summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    grouped = df.groupby(
        ["method", "algorithm", "inner_update", "inner_update_label", "inner_steps"],
        as_index=False,
    )
    summary = grouped.agg(
        seed_count=("seed", "nunique"),
        mean_rmse=("eval_rmse_raw", "mean"),
        std_rmse=("eval_rmse_raw", "std"),
        mean_mae=("eval_mae_raw", "mean"),
        std_mae=("eval_mae_raw", "std"),
        mean_best_eval_mse_norm=("best_eval_mse_norm", "mean"),
        std_best_eval_mse_norm=("best_eval_mse_norm", "std"),
        mean_time_per_step=("avg_step_time_sec", "mean"),
        std_time_per_step=("avg_step_time_sec", "std"),
        mean_meta_loss=("meta_loss", "mean"),
        std_meta_loss=("meta_loss", "std"),
    )
    summary = summary.sort_values(["mean_rmse", "mean_mae"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def combo_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    methods = [m for m in ["Second-order MAML", "FO-MAML", "Meta-SGD"] if m in set(df["method"])]
    updates = [u for u in ["supervised", "semi-supervised"] if u in set(df["inner_update"])]
    return [(method, update) for method in methods for update in updates]


def plot_seed_points(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    order = combo_order(df)
    x_positions = np.arange(len(order))
    offsets = {28: -0.12, 42: 0.0, 2026: 0.12}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    metrics = [
        ("eval_rmse_raw", "mean_rmse", "std_rmse", "Evaluation RMSE"),
        ("eval_mae_raw", "mean_mae", "std_mae", "Evaluation MAE"),
    ]

    for ax, (raw_metric, mean_metric, std_metric, ylabel) in zip(axes, metrics):
        for idx, (method, update) in enumerate(order):
            part = df[(df["method"] == method) & (df["inner_update"] == update)]
            stat = summary[(summary["method"] == method) & (summary["inner_update"] == update)]
            color = METHOD_COLORS.get(method, "#333333")
            marker = METHOD_MARKERS.get(method, "o")
            for _, row in part.iterrows():
                seed = int(row["seed"])
                offset = offsets.get(seed, (seed % 5 - 2) * 0.04)
                ax.scatter(
                    idx + offset,
                    row[raw_metric],
                    s=24,
                    color=color,
                    marker=marker,
                    alpha=0.65,
                    edgecolor="none",
                )
            if not stat.empty:
                mean = float(stat.iloc[0][mean_metric])
                std = float(stat.iloc[0][std_metric])
                ax.errorbar(
                    idx,
                    mean,
                    yerr=std,
                    fmt="D",
                    color="black",
                    markersize=4,
                    linewidth=1.0,
                    capsize=3,
                )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [
                f"{method.replace('Second-order ', '2nd ')}\n{UPDATE_LABELS[update].replace('No ', 'No ')}"
                for method, update in order
            ],
            rotation=20,
            ha="right",
        )
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = []
    labels = []
    for method in ["Second-order MAML", "FO-MAML", "Meta-SGD"]:
        if method in set(df["method"]):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=METHOD_MARKERS.get(method, "o"),
                    color="none",
                    markerfacecolor=METHOD_COLORS.get(method, "#333333"),
                    markersize=6,
                )
            )
            labels.append(method)
    handles.append(plt.Line2D([0], [0], marker="D", color="black", linestyle="none", markersize=5))
    labels.append("mean +/- SD")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Multi-seed performance on Yellow River data", y=1.04)
    save_figure(fig, output_dir / "fig_multiseed_rmse_mae")


def plot_time_points(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    order = combo_order(df)
    x_positions = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    for idx, (method, update) in enumerate(order):
        part = df[(df["method"] == method) & (df["inner_update"] == update)]
        stat = summary[(summary["method"] == method) & (summary["inner_update"] == update)]
        color = METHOD_COLORS.get(method, "#333333")
        marker = METHOD_MARKERS.get(method, "o")
        for point_idx, (_, row) in enumerate(part.iterrows()):
            offset = (point_idx - (len(part) - 1) / 2) * 0.08
            ax.scatter(
                idx + offset,
                row["avg_step_time_sec"],
                s=24,
                color=color,
                marker=marker,
                alpha=0.65,
                edgecolor="none",
            )
        if not stat.empty:
            ax.errorbar(
                idx,
                float(stat.iloc[0]["mean_time_per_step"]),
                yerr=float(stat.iloc[0]["std_time_per_step"]),
                fmt="D",
                color="black",
                markersize=4,
                linewidth=1.0,
                capsize=3,
            )

    ax.set_ylabel("Average time per step (s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            f"{method.replace('Second-order ', '2nd ')}\n{UPDATE_LABELS[update].replace('No ', 'No ')}"
            for method, update in order
        ],
        rotation=20,
        ha="right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_title("Training cost across seeds")
    save_figure(fig, output_dir / "fig_multiseed_time")


def plot_summary_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    methods = [m for m in ["Second-order MAML", "FO-MAML", "Meta-SGD"] if m in set(summary["method"])]
    updates = [u for u in ["supervised", "semi-supervised"] if u in set(summary["inner_update"])]
    matrix = summary.pivot(index="method", columns="inner_update", values="mean_rmse").reindex(methods)[updates]
    std_matrix = summary.pivot(index="method", columns="inner_update", values="std_rmse").reindex(methods)[updates]

    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), cmap="viridis_r")
    ax.set_xticks(range(len(updates)))
    ax.set_xticklabels([UPDATE_LABELS[u] for u in updates], rotation=15, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Mean RMSE across seeds")
    for row_idx, method in enumerate(methods):
        for col_idx, update in enumerate(updates):
            mean = matrix.loc[method, update]
            std = std_matrix.loc[method, update]
            ax.text(
                col_idx,
                row_idx,
                f"{mean:.3f}\n+/-{std:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Mean evaluation RMSE")
    save_figure(fig, output_dir / "fig_multiseed_mean_rmse_matrix")


def plot_results(results_path: Path, summary_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(results_path)
    summary = pd.read_csv(summary_path)
    plot_seed_points(df, summary, output_dir)
    plot_time_points(df, summary, output_dir)
    plot_summary_heatmap(summary, output_dir)


def run_experiments(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    methods = selected_methods(args.methods)
    updates = parse_list(args.inner_updates)
    for update in updates:
        if update not in UPDATE_LABELS:
            raise ValueError(f"unknown inner update: {update}")

    rows = []
    total = len(args.seeds) * len(methods) * len(updates) * len(args.inner_steps_list)
    done = 0
    for seed in args.seeds:
        args.seed = seed
        dataset = make_region_dataset(args)
        print(
            f"\nseed={seed} dataset={dataset.name}, rows={len(dataset.targets)}, "
            f"train_regions={len(dataset.train_regions)}, eval_regions={len(dataset.eval_regions)}, "
            f"features={len(dataset.feature_names)}, target={dataset.target_name}"
        )
        for method in methods:
            for update in updates:
                for inner_steps in args.inner_steps_list:
                    done += 1
                    print(f"\n=== config {done}/{total} ===")
                    row = run_one_config(args, dataset, method, update, inner_steps)
                    row["seed"] = seed
                    rows.append(row)

    results_path = output_dir / args.results_csv
    summary_path = output_dir / args.summary_csv
    write_rows(results_path, rows)
    summarize(results_path, summary_path)
    print(f"\nraw results saved to: {results_path}")
    print(f"summary saved to: {summary_path}")
    return results_path, summary_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / args.results_csv
    summary_path = output_dir / args.summary_csv

    if not args.plot_only:
        results_path, summary_path = run_experiments(args, output_dir)

    plot_results(results_path, summary_path, output_dir)
    print(f"figures saved under: {output_dir}")


if __name__ == "__main__":
    main()
