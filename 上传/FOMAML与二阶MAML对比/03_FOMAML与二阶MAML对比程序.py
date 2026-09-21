from __future__ import annotations

import argparse
import csv
import math
import os
import platform
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


def process_memory_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        pass

    if platform.system().lower() != "windows":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            return None
        return counters.WorkingSetSize / 1024 / 1024
    except Exception:
        return None


def parse_mode_list(raw: str) -> list[str]:
    allowed = {"second-order", "fomaml"}
    modes = []
    for item in raw.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in allowed:
            raise argparse.ArgumentTypeError(
                "modes must be chosen from: second-order,fomaml"
            )
        modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare second-order MAML with FOMAML create_graph=False.",
    )
    parser.add_argument("--algorithm", choices=["maml", "meta-sgd"], default="maml")
    parser.add_argument(
        "--modes",
        type=parse_mode_list,
        default=parse_mode_list("second-order,fomaml"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--meta-batch", type=int, default=8)
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
    parser.add_argument("--clamp-meta-lr", type=float, default=0.2)
    parser.add_argument("--min-meta-lr", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="fomaml_comparison_results.csv")
    return parser.parse_args()


def run_one_mode(args: argparse.Namespace, mode: str) -> dict[str, float]:
    first_order = mode == "fomaml"
    create_graph = not first_order

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

    print(
        f"\n[mode={mode}] algorithm={args.algorithm}, device={device}, "
        f"create_graph={create_graph}, inner_update={args.inner_update}, "
        f"inner_steps={args.inner_steps}, meta_batch={args.meta_batch}"
    )

    start_memory_mb = process_memory_mb()
    peak_observed_memory_mb = start_memory_mb
    started_at = time.perf_counter()
    best_val_query_loss = float("inf")
    final_row: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        task_batch = sampler.sample(args.meta_batch, device)
        metrics = meta_train_step(
            adapter=adapter,
            optimizer=optimizer,
            task_batch=task_batch,
            first_order=first_order,
            clamp_meta_lr=args.clamp_meta_lr,
            min_meta_lr=args.min_meta_lr,
            inner_update=args.inner_update,
            ssl_weight=args.ssl_weight,
            inner_steps=args.inner_steps,
        )

        current_memory_mb = process_memory_mb()
        if current_memory_mb is not None:
            if peak_observed_memory_mb is None:
                peak_observed_memory_mb = current_memory_mb
            else:
                peak_observed_memory_mb = max(
                    peak_observed_memory_mb,
                    current_memory_mb,
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
                inner_steps=args.inner_steps,
            )
            best_val_query_loss = min(best_val_query_loss, val_query_loss)
            elapsed_sec = time.perf_counter() - started_at
            val_rmse = math.sqrt(max(val_query_loss, 0.0))
            metric_text = " ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            )
            memory_text = ""
            if current_memory_mb is not None:
                memory_text = f" process_memory_mb={current_memory_mb:.2f}"
            print(
                f"step={step:04d} {metric_text} "
                f"val_query_loss={val_query_loss:.6f} "
                f"val_rmse={val_rmse:.6f} "
                f"best_val_query_loss={best_val_query_loss:.6f} "
                f"elapsed_sec={elapsed_sec:.2f}"
                f"{memory_text}"
            )
            final_row = {
                "mode": mode,
                "algorithm": args.algorithm,
                "create_graph": create_graph,
                "first_order": first_order,
                "inner_update": args.inner_update,
                "inner_steps": args.inner_steps,
                "meta_batch": args.meta_batch,
                "step": step,
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
                "val_rmse": val_rmse,
                "best_val_query_loss": best_val_query_loss,
                "elapsed_sec": elapsed_sec,
                "avg_step_time_sec": elapsed_sec / step,
                "start_process_memory_mb": start_memory_mb,
                "end_process_memory_mb": current_memory_mb,
                "peak_observed_process_memory_mb": peak_observed_memory_mb,
                **metrics,
            }

    if device.type == "cuda" and final_row:
        final_row["cuda_peak_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
        )

    return final_row


def write_results(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "algorithm",
        "create_graph",
        "first_order",
        "inner_update",
        "inner_steps",
        "meta_batch",
        "step",
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
        "val_rmse",
        "best_val_query_loss",
        "grad_norm",
        "elapsed_sec",
        "avg_step_time_sec",
        "start_process_memory_mb",
        "end_process_memory_mb",
        "peak_observed_process_memory_mb",
        "cuda_peak_memory_mb",
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
    for mode in args.modes:
        rows.append(run_one_mode(args, mode))

    output_path = Path(args.output)
    write_results(output_path, rows)

    print("\nsummary")
    print(
        "mode,create_graph,val_query_loss,val_rmse,best_val_query_loss,"
        "meta_loss,grad_norm,avg_step_time_sec,peak_observed_process_memory_mb,"
        "cuda_peak_memory_mb"
    )
    for row in rows:
        print(
            f"{row['mode']},"
            f"{row['create_graph']},"
            f"{row['val_query_loss']:.6f},"
            f"{row['val_rmse']:.6f},"
            f"{row['best_val_query_loss']:.6f},"
            f"{row['meta_loss']:.6f},"
            f"{row['grad_norm']:.6f},"
            f"{row['avg_step_time_sec']:.4f},"
            f"{row.get('peak_observed_process_memory_mb', '')},"
            f"{row.get('cuda_peak_memory_mb', '')}"
        )
    print(f"\nresults saved to: {output_path}")


if __name__ == "__main__":
    main()
