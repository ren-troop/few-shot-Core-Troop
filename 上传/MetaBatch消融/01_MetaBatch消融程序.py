from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from maml_meta_sgd_framework import (
    FewShotTaskSampler,
    build_adapter,
    choose_device,
    evaluate,
    meta_train_step,
)


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("meta batch values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one meta batch value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run meta-batch-size ablation for MAML / Meta-SGD.",
    )
    parser.add_argument("--algorithm", choices=["maml", "meta-sgd"], default="meta-sgd")
    parser.add_argument(
        "--meta-batch-list",
        type=parse_int_list,
        default=parse_int_list("2,8,16"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--total-task-budget",
        type=int,
        default=0,
        help=(
            "If > 0, each meta batch setting uses approximately the same total "
            "number of sampled training tasks instead of the same outer steps."
        ),
    )
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--unlabeled-per-task", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument(
        "--inner-update",
        choices=["supervised", "semi-supervised"],
        default="semi-supervised",
    )
    parser.add_argument("--ssl-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-noise", type=float, default=0.1)
    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--clamp-meta-lr", type=float, default=0.2)
    parser.add_argument("--min-meta-lr", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="meta_batch_results.csv")
    return parser.parse_args()


def effective_steps(args: argparse.Namespace, meta_batch: int) -> int:
    if args.total_task_budget <= 0:
        return args.steps
    return max(1, math.ceil(args.total_task_budget / meta_batch))


def run_one_setting(args: argparse.Namespace, meta_batch: int) -> dict[str, float]:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sampler = FewShotTaskSampler(
        shots=args.shots,
        queries=args.queries,
        unlabeled_per_task=args.unlabeled_per_task,
        pseudo_noise=args.pseudo_noise,
    )
    adapter = build_adapter(args, device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.outer_lr)
    steps = effective_steps(args, meta_batch)

    print(
        f"\n[meta_batch={meta_batch}] algorithm={args.algorithm}, device={device}, "
        f"inner_update={args.inner_update}, inner_steps={args.inner_steps}, "
        f"first_order={args.first_order}, steps={steps}"
    )

    started_at = time.perf_counter()
    best_val_query_loss = float("inf")
    final_row: dict[str, float] = {}

    for step in range(1, steps + 1):
        task_batch = sampler.sample(meta_batch, device)
        metrics = meta_train_step(
            adapter=adapter,
            optimizer=optimizer,
            task_batch=task_batch,
            first_order=args.first_order,
            clamp_meta_lr=args.clamp_meta_lr,
            min_meta_lr=args.min_meta_lr,
            inner_update=args.inner_update,
            ssl_weight=args.ssl_weight,
            inner_steps=args.inner_steps,
        )

        should_log = step == 1 or step % args.log_every == 0 or step == steps
        if should_log:
            val_query_loss = evaluate(
                adapter=adapter,
                sampler=sampler,
                val_tasks=args.val_tasks,
                device=device,
                inner_update=args.inner_update,
                ssl_weight=args.ssl_weight,
                inner_steps=args.inner_steps,
            )
            best_val_query_loss = min(best_val_query_loss, val_query_loss)
            elapsed_sec = time.perf_counter() - started_at
            metric_text = " ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            )
            print(
                f"step={step:04d} {metric_text} "
                f"val_query_loss={val_query_loss:.6f} "
                f"best_val_query_loss={best_val_query_loss:.6f} "
                f"elapsed_sec={elapsed_sec:.2f}"
            )
            final_row = {
                "algorithm": args.algorithm,
                "inner_update": args.inner_update,
                "inner_steps": args.inner_steps,
                "step": step,
                "configured_steps": steps,
                "total_task_budget": args.total_task_budget,
                "first_order": args.first_order,
                "meta_batch": meta_batch,
                "sampled_train_tasks": step * meta_batch,
                "shots": args.shots,
                "queries": args.queries,
                "unlabeled_per_task": args.unlabeled_per_task,
                "ssl_weight": args.ssl_weight,
                "pseudo_noise": args.pseudo_noise,
                "outer_lr": args.outer_lr,
                "inner_lr": args.inner_lr,
                "min_meta_lr": args.min_meta_lr,
                "clamp_meta_lr": args.clamp_meta_lr,
                "val_query_loss": val_query_loss,
                "best_val_query_loss": best_val_query_loss,
                "elapsed_sec": elapsed_sec,
                "avg_step_time_sec": elapsed_sec / step,
                **metrics,
            }

    if device.type == "cuda" and final_row:
        peak_bytes = torch.cuda.max_memory_allocated(device)
        final_row["cuda_peak_memory_mb"] = peak_bytes / 1024 / 1024

    return final_row


def write_results(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "inner_update",
        "inner_steps",
        "meta_batch",
        "step",
        "configured_steps",
        "sampled_train_tasks",
        "total_task_budget",
        "first_order",
        "shots",
        "queries",
        "unlabeled_per_task",
        "ssl_weight",
        "pseudo_noise",
        "outer_lr",
        "inner_lr",
        "min_meta_lr",
        "clamp_meta_lr",
        "inner_loss",
        "support_loss",
        "ssl_loss",
        "meta_loss",
        "val_query_loss",
        "best_val_query_loss",
        "grad_norm",
        "elapsed_sec",
        "avg_step_time_sec",
        "meta_lr_mean",
        "meta_lr_min",
        "meta_lr_max",
        "meta_lr_nonpositive_count",
        "cuda_peak_memory_mb",
    ]
    seen = set()
    deduped_fieldnames = []
    for name in fieldnames:
        if name not in seen:
            deduped_fieldnames.append(name)
            seen.add(name)
    for row in rows:
        for key in row:
            if key not in seen:
                deduped_fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=deduped_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for meta_batch in args.meta_batch_list:
        rows.append(run_one_setting(args, meta_batch))

    output_path = Path(args.output)
    write_results(output_path, rows)

    print("\nsummary")
    print(
        "meta_batch,steps,sampled_train_tasks,val_query_loss,"
        "best_val_query_loss,meta_loss,grad_norm,avg_step_time_sec"
    )
    for row in rows:
        print(
            f"{row['meta_batch']},"
            f"{row['configured_steps']},"
            f"{row['sampled_train_tasks']},"
            f"{row['val_query_loss']:.6f},"
            f"{row['best_val_query_loss']:.6f},"
            f"{row['meta_loss']:.6f},"
            f"{row['grad_norm']:.6f},"
            f"{row['avg_step_time_sec']:.4f}"
        )
    print(f"\nresults saved to: {output_path}")


if __name__ == "__main__":
    main()
