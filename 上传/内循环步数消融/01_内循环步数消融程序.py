from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from maml_meta_sgd_framework import (
    FewShotTaskSampler,
    build_adapter,
    choose_device,
    evaluate,
    meta_train_step,
)


def parse_step_list(raw: str) -> list[int]:
    steps = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("inner step values must be positive")
        steps.append(value)
    if not steps:
        raise argparse.ArgumentTypeError("at least one inner step value is required")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inner-loop step ablation for MAML / Meta-SGD.",
    )
    parser.add_argument("--algorithm", choices=["maml", "meta-sgd"], default="meta-sgd")
    parser.add_argument(
        "--inner-steps-list",
        type=parse_step_list,
        default=parse_step_list("1,3,5"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--unlabeled-per-task", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--inner-lr", type=float, default=0.01)
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
    parser.add_argument("--output", default="inner_steps_results.csv")
    return parser.parse_args()


def run_one_setting(args: argparse.Namespace, inner_steps: int) -> dict[str, float]:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    sampler = FewShotTaskSampler(
        shots=args.shots,
        queries=args.queries,
        unlabeled_per_task=args.unlabeled_per_task,
        pseudo_noise=args.pseudo_noise,
    )
    adapter = build_adapter(args, device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.outer_lr)

    print(
        f"\n[inner_steps={inner_steps}] algorithm={args.algorithm}, device={device}, "
        f"inner_update={args.inner_update}, first_order={args.first_order}, "
        f"meta_batch={args.meta_batch}"
    )

    best_val_query_loss = float("inf")
    final_row: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        task_batch = sampler.sample(args.meta_batch, device)
        metrics = meta_train_step(
            adapter=adapter,
            optimizer=optimizer,
            task_batch=task_batch,
            first_order=args.first_order,
            clamp_meta_lr=args.clamp_meta_lr,
            min_meta_lr=args.min_meta_lr,
            inner_update=args.inner_update,
            ssl_weight=args.ssl_weight,
            inner_steps=inner_steps,
        )

        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        if should_log:
            val_query_loss = evaluate(
                adapter=adapter,
                sampler=sampler,
                val_tasks=args.val_tasks,
                device=device,
                inner_update=args.inner_update,
                ssl_weight=args.ssl_weight,
                inner_steps=inner_steps,
            )
            best_val_query_loss = min(best_val_query_loss, val_query_loss)
            metric_text = " ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            )
            print(
                f"step={step:04d} {metric_text} "
                f"val_query_loss={val_query_loss:.6f} "
                f"best_val_query_loss={best_val_query_loss:.6f}"
            )
            final_row = {
                "algorithm": args.algorithm,
                "inner_update": args.inner_update,
                "inner_steps": inner_steps,
                "step": step,
                "first_order": args.first_order,
                "meta_batch": args.meta_batch,
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
                **metrics,
            }

    return final_row


def write_results(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "inner_update",
        "inner_steps",
        "step",
        "first_order",
        "meta_batch",
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
        "inner_lr",
        "meta_lr_mean",
        "meta_lr_min",
        "meta_lr_max",
        "meta_lr_nonpositive_count",
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
    for inner_steps in args.inner_steps_list:
        rows.append(run_one_setting(args, inner_steps))

    output_path = Path(args.output)
    write_results(output_path, rows)

    print("\nsummary")
    print("inner_steps,val_query_loss,best_val_query_loss,meta_loss,grad_norm")
    for row in rows:
        print(
            f"{row['inner_steps']},"
            f"{row['val_query_loss']:.6f},"
            f"{row['best_val_query_loss']:.6f},"
            f"{row['meta_loss']:.6f},"
            f"{row['grad_norm']:.6f}"
        )
    print(f"\nresults saved to: {output_path}")


if __name__ == "__main__":
    main()
