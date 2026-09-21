from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from public_region_maml_meta_sgd import (
    RegionalTaskSampler,
    build_adapter,
    choose_device,
    evaluate,
    make_region_dataset,
    meta_train_step,
    parse_list,
)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    algorithm: str
    first_order: bool
    create_graph: bool


METHODS = [
    MethodSpec(
        label="Second-order MAML",
        algorithm="maml",
        first_order=False,
        create_graph=True,
    ),
    MethodSpec(
        label="FO-MAML",
        algorithm="maml",
        first_order=True,
        create_graph=False,
    ),
    MethodSpec(
        label="Meta-SGD",
        algorithm="meta-sgd",
        first_order=False,
        create_graph=True,
    ),
]

UPDATE_LABELS = {
    "supervised": "No semi-supervised",
    "semi-supervised": "Semi-supervised",
}

METHOD_COLORS = {
    "Second-order MAML": "#0072B2",
    "FO-MAML": "#E69F00",
    "Meta-SGD": "#009E73",
}

METHOD_MARKERS = {
    "Second-order MAML": "o",
    "FO-MAML": "s",
    "Meta-SGD": "^",
}


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("inner step values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one inner step value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full comparison: second-order MAML vs FO-MAML vs Meta-SGD, "
            "with/without semi-supervised inner loop, across inner steps."
        )
    )
    parser.add_argument("--dataset", choices=["california", "csv"], default="california")
    parser.add_argument("--data-dir", default="sklearn_data")
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--region-column", default="region_id")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--feature-columns", default="")
    parser.add_argument("--drop-columns", default="")
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
        help="Comma-separated update modes.",
    )
    parser.add_argument(
        "--inner-steps-list",
        type=parse_int_list,
        default=parse_int_list("1,3,5"),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--unlabeled-per-task", type=int, default=16)
    parser.add_argument("--eval-tasks", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--min-meta-lr", type=float, default=1e-6)
    parser.add_argument("--clamp-meta-lr", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="full_comparison_outputs")
    parser.add_argument("--results-csv", default="full_comparison_results.csv")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def selected_methods(raw: str) -> list[MethodSpec]:
    requested = parse_list(raw)
    method_by_label = {method.label: method for method in METHODS}
    missing = [label for label in requested if label not in method_by_label]
    if missing:
        raise ValueError(f"unknown method labels: {missing}")
    return [method_by_label[label] for label in requested]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
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
    seen = set()
    fieldnames = []
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


def run_one_config(
    args: argparse.Namespace,
    dataset,
    method: MethodSpec,
    inner_update: str,
    inner_steps: int,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_sampler = RegionalTaskSampler(
        dataset=dataset,
        split="train",
        shots=args.shots,
        queries=args.queries,
        unlabeled_per_task=args.unlabeled_per_task,
        seed=args.seed,
    )
    eval_sampler = RegionalTaskSampler(
        dataset=dataset,
        split="eval",
        shots=args.shots,
        queries=args.queries,
        unlabeled_per_task=args.unlabeled_per_task,
        seed=args.seed + 1000,
    )
    adapter = build_adapter(
        algorithm=method.algorithm,
        input_dim=dataset.features.shape[1],
        hidden_dim=args.hidden_dim,
        inner_lr=args.inner_lr,
        device=device,
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.outer_lr)

    ssl_weight_used = args.ssl_weight if inner_update == "semi-supervised" else 0.0
    best_eval_mse_norm = float("inf")
    started_at = time.perf_counter()
    final_row: dict[str, object] = {}

    print(
        f"\n[{method.label} | {UPDATE_LABELS[inner_update]} | inner_steps={inner_steps}] "
        f"algorithm={method.algorithm}, create_graph={method.create_graph}, "
        f"meta_batch={args.meta_batch}, steps={args.steps}"
    )

    for step in range(1, args.steps + 1):
        task_batch = train_sampler.sample(args.meta_batch, device)
        train_metrics = meta_train_step(
            adapter=adapter,
            optimizer=optimizer,
            task_batch=task_batch,
            first_order=method.first_order,
            inner_update=inner_update,
            ssl_weight=ssl_weight_used,
            inner_steps=inner_steps,
            min_meta_lr=args.min_meta_lr,
            clamp_meta_lr=args.clamp_meta_lr,
        )

        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        if should_log:
            eval_metrics = evaluate(
                adapter=adapter,
                sampler=eval_sampler,
                val_tasks=args.eval_tasks,
                device=device,
                inner_update=inner_update,
                ssl_weight=ssl_weight_used,
                inner_steps=inner_steps,
                target_mean=dataset.target_mean,
                target_std=dataset.target_std,
            )
            best_eval_mse_norm = min(best_eval_mse_norm, eval_metrics["eval_mse_norm"])
            elapsed_sec = time.perf_counter() - started_at
            metric_text = (
                f"meta_loss={train_metrics['meta_loss']:.6f} "
                f"eval_rmse_raw={eval_metrics['eval_rmse_raw']:.6f} "
                f"eval_mae_raw={eval_metrics['eval_mae_raw']:.6f} "
                f"best_eval_mse_norm={best_eval_mse_norm:.6f} "
                f"elapsed_sec={elapsed_sec:.2f}"
            )
            print(f"step={step:04d} {metric_text}")

            final_row = {
                "method": method.label,
                "algorithm": method.algorithm,
                "create_graph": method.create_graph,
                "first_order": method.first_order,
                "inner_update": inner_update,
                "inner_update_label": UPDATE_LABELS[inner_update],
                "inner_steps": inner_steps,
                "step": step,
                "dataset": dataset.name,
                "train_regions": len(dataset.train_regions),
                "eval_regions": len(dataset.eval_regions),
                "feature_count": len(dataset.feature_names),
                "target": dataset.target_name,
                "meta_batch": args.meta_batch,
                "shots": args.shots,
                "queries": args.queries,
                "unlabeled_per_task": args.unlabeled_per_task,
                "ssl_weight_used": ssl_weight_used,
                "outer_lr": args.outer_lr,
                "inner_lr": args.inner_lr,
                "best_eval_mse_norm": best_eval_mse_norm,
                "elapsed_sec": elapsed_sec,
                "avg_step_time_sec": elapsed_sec / step,
                **train_metrics,
                **eval_metrics,
            }

    if device.type == "cuda" and final_row:
        final_row["cuda_peak_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
        )

    return final_row


def run_experiments(args: argparse.Namespace, output_dir: Path) -> Path:
    dataset = make_region_dataset(args)
    methods = selected_methods(args.methods)
    updates = parse_list(args.inner_updates)
    for update in updates:
        if update not in UPDATE_LABELS:
            raise ValueError(f"unknown inner update: {update}")

    print(
        f"loaded dataset={dataset.name}, rows={len(dataset.targets)}, "
        f"train_regions={len(dataset.train_regions)}, eval_regions={len(dataset.eval_regions)}, "
        f"features={dataset.feature_names}, target={dataset.target_name}"
    )

    rows = []
    total = len(methods) * len(updates) * len(args.inner_steps_list)
    done = 0
    for method in methods:
        for inner_update in updates:
            for inner_steps in args.inner_steps_list:
                done += 1
                print(f"\n=== config {done}/{total} ===")
                rows.append(run_one_config(args, dataset, method, inner_update, inner_steps))

    results_path = output_dir / args.results_csv
    write_rows(results_path, rows)
    print(f"\nall results saved to: {results_path}")
    return results_path


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_metric_grid(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_base: Path,
) -> None:
    update_order = ["supervised", "semi-supervised"]
    method_order = [method.label for method in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)

    for ax, update in zip(axes, update_order):
        part = df[df["inner_update"] == update]
        for method in method_order:
            method_part = part[part["method"] == method].sort_values("inner_steps")
            if method_part.empty:
                continue
            ax.plot(
                method_part["inner_steps"],
                method_part[metric],
                label=method,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=1.8,
                markersize=5,
            )
        ax.set_title(UPDATE_LABELS[update])
        ax.set_xlabel("Inner-loop steps")
        ax.set_xticks(sorted(df["inner_steps"].unique()))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, linestyle="--")
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(title, y=0.98, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, output_base: Path) -> None:
    method_order = [method.label for method in METHODS]
    update_markers = {
        "supervised": "o",
        "semi-supervised": "s",
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for method in method_order:
        for update, marker in update_markers.items():
            part = df[(df["method"] == method) & (df["inner_update"] == update)]
            if part.empty:
                continue
            ax.scatter(
                part["avg_step_time_sec"],
                part["eval_rmse_raw"],
                s=30 + part["inner_steps"] * 16,
                color=METHOD_COLORS[method],
                marker=marker,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.82,
                label=f"{method} | {UPDATE_LABELS[update]}",
            )
            for _, row in part.iterrows():
                ax.text(
                    row["avg_step_time_sec"],
                    row["eval_rmse_raw"],
                    str(int(row["inner_steps"])),
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="white",
                )

    ax.set_xlabel("Average time per step (s)")
    ax.set_ylabel("Evaluation RMSE (raw scale)")
    ax.set_title("Accuracy-speed tradeoff", fontweight="bold")
    ax.grid(color="#D9D9D9", linewidth=0.6, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_like(df: pd.DataFrame, output_base: Path) -> None:
    updates = ["supervised", "semi-supervised"]
    method_order = [method.label for method in METHODS]
    steps = sorted(df["inner_steps"].unique())
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    vmin = df["eval_rmse_raw"].min()
    vmax = df["eval_rmse_raw"].max()

    for ax, update in zip(axes, updates):
        matrix = []
        for method in method_order:
            row = []
            for step in steps:
                value = df[
                    (df["inner_update"] == update)
                    & (df["method"] == method)
                    & (df["inner_steps"] == step)
                ]["eval_rmse_raw"]
                row.append(float(value.iloc[0]) if not value.empty else float("nan"))
            matrix.append(row)

        im = ax.imshow(matrix, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(UPDATE_LABELS[update])
        ax.set_xticks(range(len(steps)), labels=[str(step) for step in steps])
        ax.set_yticks(range(len(method_order)), labels=method_order)
        ax.set_xlabel("Inner-loop steps")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > (vmin + vmax) / 2 else "black",
                )

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Evaluation RMSE (raw scale)")
    fig.suptitle("RMSE matrix across methods and inner-loop steps", fontweight="bold")
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_results(results_path: Path, output_dir: Path) -> None:
    setup_plot_style()
    df = pd.read_csv(results_path)
    plot_metric_grid(
        df,
        metric="eval_rmse_raw",
        ylabel="Evaluation RMSE (raw scale)",
        title="Cross-region prediction accuracy",
        output_base=output_dir / "fig_accuracy_rmse",
    )
    plot_metric_grid(
        df,
        metric="avg_step_time_sec",
        ylabel="Average time per step (s)",
        title="Training speed comparison",
        output_base=output_dir / "fig_speed",
    )
    plot_metric_grid(
        df,
        metric="meta_loss",
        ylabel="Final meta loss",
        title="Final meta-training loss",
        output_base=output_dir / "fig_meta_loss",
    )
    plot_tradeoff(df, output_dir / "fig_accuracy_speed_tradeoff")
    plot_heatmap_like(df, output_dir / "fig_rmse_matrix")
    print(f"figures saved under: {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / args.results_csv

    if not args.plot_only:
        results_path = run_experiments(args, output_dir)
    elif not results_path.exists():
        raise FileNotFoundError(results_path)

    plot_results(results_path, output_dir)


if __name__ == "__main__":
    main()
