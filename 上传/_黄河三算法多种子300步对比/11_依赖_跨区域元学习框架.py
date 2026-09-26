from __future__ import annotations

import argparse
import csv
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TaskBatch:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    ssl_x: torch.Tensor
    ssl_y: torch.Tensor
    ssl_confidence: torch.Tensor


@dataclass
class RegionDataset:
    name: str
    features: np.ndarray
    targets: np.ndarray
    regions: np.ndarray
    train_regions: list[str]
    eval_regions: list[str]
    feature_names: list[str]
    target_name: str
    target_mean: float
    target_std: float


class RegionMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),    #激活函数，\(\text{ReLU}(x) = \max(0, x)\)正数保持不变，负数直接置为 0。
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        params: OrderedDict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            return self.net(x)

        x = F.linear(x, params["net.0.weight"], params["net.0.bias"])
        x = F.relu(x)
        x = F.linear(x, params["net.2.weight"], params["net.2.bias"])
        x = F.relu(x)
        x = F.linear(x, params["net.4.weight"], params["net.4.bias"])
        return x


class MAMLAdapter(nn.Module):
    def __init__(self, model: RegionMLPRegressor, inner_lr: float) -> None:
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        ssl_x: torch.Tensor,
        ssl_y: torch.Tensor,
        ssl_confidence: torch.Tensor,
        inner_update: str,
        ssl_weight: float,
        params: OrderedDict[str, torch.Tensor] | None,
        create_graph: bool,
    ) -> tuple[OrderedDict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if params is None:
            params = OrderedDict(self.model.named_parameters())

        inner_loss, support_loss, ssl_loss = compute_inner_loss(
            self.model,
            params,
            support_x,
            support_y,
            ssl_x,
            ssl_y,
            ssl_confidence,
            inner_update,
            ssl_weight,
        )
        grads = torch.autograd.grad(
            inner_loss,
            params.values(),
            create_graph=create_graph,
        )
        adapted_params = OrderedDict(
            (name, param - self.inner_lr * grad)
            for (name, param), grad in zip(params.items(), grads)
        )
        return adapted_params, inner_loss, support_loss, ssl_loss

    def extra_metrics(self) -> dict[str, float]:
        return {"inner_lr": self.inner_lr}


class MetaSGDAdapter(nn.Module):
    def __init__(self, model: RegionMLPRegressor, init_inner_lr: float) -> None:
        super().__init__()
        self.model = model
        self.meta_lrs = nn.ParameterList(
            [
                nn.Parameter(torch.full_like(param, init_inner_lr))
                for param in model.parameters()
            ]
        )

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        ssl_x: torch.Tensor,
        ssl_y: torch.Tensor,
        ssl_confidence: torch.Tensor,
        inner_update: str,
        ssl_weight: float,
        params: OrderedDict[str, torch.Tensor] | None,
        create_graph: bool,
    ) -> tuple[OrderedDict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if params is None:
            params = OrderedDict(self.model.named_parameters())

        inner_loss, support_loss, ssl_loss = compute_inner_loss(
            self.model,
            params,
            support_x,
            support_y,
            ssl_x,
            ssl_y,
            ssl_confidence,
            inner_update,
            ssl_weight,
        )
        grads = torch.autograd.grad(
            inner_loss,
            params.values(),
            create_graph=create_graph,
        )
        adapted_params = OrderedDict(
            (name, param - meta_lr * grad)
            for (name, param), meta_lr, grad in zip(params.items(), self.meta_lrs, grads)
        )
        return adapted_params, inner_loss, support_loss, ssl_loss

    def clamp_meta_lrs(self, min_value: float, max_value: float) -> None:
        lower = max(min_value, 1e-12)
        upper = max(max_value, lower)
        with torch.no_grad():
            for meta_lr in self.meta_lrs:
                meta_lr.clamp_(min=lower, max=upper)

    def extra_metrics(self) -> dict[str, float]:
        flat_lrs = torch.cat([meta_lr.detach().flatten() for meta_lr in self.meta_lrs])
        return {
            "meta_lr_mean": flat_lrs.mean().item(),
            "meta_lr_min": flat_lrs.min().item(),
            "meta_lr_max": flat_lrs.max().item(),
            "meta_lr_nonpositive_count": (flat_lrs <= 0).sum().item(),
        }


class RegionalTaskSampler:
    def __init__(
        self,
        dataset: RegionDataset,
        split: str,
        shots: int,
        queries: int,
        unlabeled_per_task: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.shots = shots
        self.queries = queries
        self.unlabeled_per_task = unlabeled_per_task
        self.rng = np.random.default_rng(seed)

        if split == "train":
            selected_regions = dataset.train_regions
        elif split == "eval":
            selected_regions = dataset.eval_regions
        else:
            raise ValueError("split must be train or eval")

        self.region_to_indices = {
            region: np.flatnonzero(dataset.regions == region)
            for region in selected_regions
        }
        self.region_to_indices = {
            region: indices
            for region, indices in self.region_to_indices.items()
            if len(indices) >= max(2, shots + queries)
        }
        self.regions = sorted(self.region_to_indices)
        if not self.regions:
            raise ValueError(f"no usable regions for split={split}")

    def sample(self, num_tasks: int, device: torch.device) -> TaskBatch:
        task_regions = self.rng.choice(self.regions, size=num_tasks, replace=True)

        support_x, support_y = [], []
        query_x, query_y = [], []
        ssl_x, ssl_y, ssl_confidence = [], [], []

        for region in task_regions:
            indices = self.region_to_indices[region]
            needed = self.shots + self.queries + self.unlabeled_per_task
            replace = len(indices) < needed
            sampled = self.rng.choice(indices, size=needed, replace=replace)

            support_idx = sampled[: self.shots]
            query_idx = sampled[self.shots : self.shots + self.queries]
            unlabeled_idx = sampled[self.shots + self.queries :]

            region_support_x = self.dataset.features[support_idx]
            region_support_y = self.dataset.targets[support_idx]
            region_query_x = self.dataset.features[query_idx]
            region_query_y = self.dataset.targets[query_idx]
            region_ssl_x = self.dataset.features[unlabeled_idx]

            region_ssl_y, region_conf = self.make_pseudo_labels(
                region_support_x,
                region_support_y,
                region_ssl_x,
            )

            support_x.append(region_support_x)
            support_y.append(region_support_y)
            query_x.append(region_query_x)
            query_y.append(region_query_y)
            ssl_x.append(region_ssl_x)
            ssl_y.append(region_ssl_y)
            ssl_confidence.append(region_conf)

        return TaskBatch(
            to_tensor(np.stack(support_x), device),
            to_tensor(np.stack(support_y), device),
            to_tensor(np.stack(query_x), device),
            to_tensor(np.stack(query_y), device),
            to_tensor(np.stack(ssl_x), device),
            to_tensor(np.stack(ssl_y), device),
            to_tensor(np.stack(ssl_confidence), device),
        )

    @staticmethod
    def make_pseudo_labels(
        support_x: np.ndarray,
        support_y: np.ndarray,
        unlabeled_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(unlabeled_x) == 0:
            return (
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )

        diff = unlabeled_x[:, None, :] - support_x[None, :, :]
        distances = np.sqrt(np.mean(diff * diff, axis=2))
        nearest = distances.argmin(axis=1)
        nearest_dist = distances[np.arange(len(unlabeled_x)), nearest]
        scale = np.median(distances) + 1e-6
        confidence = np.exp(-nearest_dist / scale).reshape(-1, 1)
        pseudo_y = support_y[nearest].reshape(-1, 1)
        return pseudo_y.astype(np.float32), confidence.astype(np.float32)


def to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def compute_inner_loss(
    model: RegionMLPRegressor,
    params: OrderedDict[str, torch.Tensor],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    ssl_x: torch.Tensor,
    ssl_y: torch.Tensor,
    ssl_confidence: torch.Tensor,
    inner_update: str,
    ssl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_loss = F.mse_loss(model(support_x, params), support_y)
    if inner_update != "semi-supervised" or ssl_x.numel() == 0 or ssl_weight <= 0:
        return support_loss, support_loss, support_loss.new_zeros(())

    ssl_pred = model(ssl_x, params)
    ssl_loss_per_sample = F.mse_loss(ssl_pred, ssl_y, reduction="none")
    weighted_ssl_loss = (
        ssl_loss_per_sample * ssl_confidence
    ).sum() / ssl_confidence.sum().clamp_min(1.0)
    return support_loss + ssl_weight * weighted_ssl_loss, support_loss, weighted_ssl_loss


def total_grad_norm(parameters: Iterable[torch.Tensor]) -> float:
    squared_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            squared_norm += param.grad.detach().pow(2).sum().item()
    return squared_norm ** 0.5


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_adapter(
    algorithm: str,
    input_dim: int,
    hidden_dim: int,
    inner_lr: float,
    device: torch.device,
) -> MAMLAdapter | MetaSGDAdapter:
    model = RegionMLPRegressor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    if algorithm == "maml":
        return MAMLAdapter(model, inner_lr=inner_lr).to(device)
    if algorithm == "meta-sgd":
        return MetaSGDAdapter(model, init_inner_lr=inner_lr).to(device)
    raise ValueError(f"unknown algorithm: {algorithm}")


def load_california_housing(args: argparse.Namespace) -> tuple[pd.DataFrame, str, str, list[str]]:
    from sklearn.datasets import fetch_california_housing

    data_home = Path(args.data_dir)
    data_home.mkdir(parents=True, exist_ok=True)
    try:
        housing = fetch_california_housing(as_frame=True, data_home=str(data_home))
    except Exception as exc:
        raise RuntimeError(
            "California Housing 数据集未能加载。请先联网运行一次，或改用 "
            "--dataset csv --csv-path <你的CSV路径>。"
        ) from exc

    df = housing.frame.copy()
    target_column = "MedHouseVal"

    lat_bin = pd.qcut(
        df["Latitude"],
        q=args.region_bins,
        labels=False,
        duplicates="drop",
    )
    lon_bin = pd.qcut(
        df["Longitude"],
        q=args.region_bins,
        labels=False,
        duplicates="drop",
    )
    df["region_id"] = "lat" + lat_bin.astype(str) + "_lon" + lon_bin.astype(str)

    feature_columns = list(housing.feature_names)
    return df, "region_id", target_column, feature_columns


def load_csv_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, str, str, list[str]]:
    if not args.csv_path:
        raise ValueError("--csv-path is required when --dataset csv")

    df = pd.read_csv(args.csv_path)
    if args.region_column not in df.columns:
        raise ValueError(f"region column not found: {args.region_column}")
    if args.target_column not in df.columns:
        raise ValueError(f"target column not found: {args.target_column}")

    if args.feature_columns:
        feature_columns = parse_list(args.feature_columns)
    else:
        blocked = {args.region_column, args.target_column}
        blocked.update(parse_list(args.drop_columns))
        feature_columns = [
            column
            for column in df.columns
            if column not in blocked and pd.api.types.is_numeric_dtype(df[column])
        ]

    if not feature_columns:
        raise ValueError("no feature columns selected")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"feature columns not found: {missing}")

    return df, args.region_column, args.target_column, feature_columns


def make_region_dataset(args: argparse.Namespace) -> RegionDataset:
    if args.dataset == "california":
        df, region_column, target_column, feature_columns = load_california_housing(args)
        dataset_name = "California Housing"
    elif args.dataset == "csv":
        df, region_column, target_column, feature_columns = load_csv_dataset(args)
        dataset_name = f"CSV:{Path(args.csv_path).name}"
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    used_columns = [region_column, target_column, *feature_columns]
    df = df[used_columns].replace([np.inf, -np.inf], np.nan).dropna().copy()
    counts = df.groupby(region_column).size()
    valid_regions = counts[counts >= args.min_samples_per_region].index
    df = df[df[region_column].isin(valid_regions)].copy()
    if df.empty:
        raise ValueError("no data left after filtering regions")

    rng = np.random.default_rng(args.seed)
    regions = sorted(df[region_column].astype(str).unique())
    if len(regions) < 2:
        raise ValueError(
            "cross-region experiment requires at least 2 usable regions/tasks"
        )
    rng.shuffle(regions)
    eval_count = max(1, int(round(len(regions) * args.eval_region_fraction)))
    eval_count = min(eval_count, len(regions) - 1)
    eval_regions = sorted(regions[:eval_count])
    train_regions = sorted(regions[eval_count:])

    row_regions = df[region_column].astype(str).to_numpy()
    train_mask = np.isin(row_regions, train_regions)

    features = df[feature_columns].astype("float32").to_numpy()
    targets = df[target_column].astype("float32").to_numpy().reshape(-1, 1)

    feature_mean = features[train_mask].mean(axis=0, keepdims=True)
    feature_std = features[train_mask].std(axis=0, keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    target_mean = float(targets[train_mask].mean())
    target_std = float(targets[train_mask].std())
    if target_std < 1e-6:
        target_std = 1.0

    norm_features = (features - feature_mean) / feature_std
    norm_targets = (targets - target_mean) / target_std

    return RegionDataset(
        name=dataset_name,
        features=norm_features.astype("float32"),
        targets=norm_targets.astype("float32"),
        regions=row_regions,
        train_regions=train_regions,
        eval_regions=eval_regions,
        feature_names=feature_columns,
        target_name=target_column,
        target_mean=target_mean,
        target_std=target_std,
    )


def meta_train_step(
    adapter: MAMLAdapter | MetaSGDAdapter,
    optimizer: torch.optim.Optimizer,
    task_batch: TaskBatch,
    first_order: bool,
    inner_update: str,
    ssl_weight: float,
    inner_steps: int,
    min_meta_lr: float,
    clamp_meta_lr: float,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    task_count = task_batch.support_x.shape[0]
    meta_loss = torch.zeros((), device=task_batch.support_x.device)
    inner_loss_sum = 0.0
    support_loss_sum = 0.0
    ssl_loss_sum = 0.0

    for task_idx in range(task_count):
        adapted_params = None
        inner_loss = torch.zeros((), device=task_batch.support_x.device)
        support_loss = torch.zeros((), device=task_batch.support_x.device)
        ssl_loss = torch.zeros((), device=task_batch.support_x.device)

        for _ in range(inner_steps):
            adapted_params, inner_loss, support_loss, ssl_loss = adapter.adapt(
                task_batch.support_x[task_idx],
                task_batch.support_y[task_idx],
                task_batch.ssl_x[task_idx],
                task_batch.ssl_y[task_idx],
                task_batch.ssl_confidence[task_idx],
                inner_update=inner_update,
                ssl_weight=ssl_weight,
                params=adapted_params,
                create_graph=not first_order,
            )

        query_pred = adapter.model(task_batch.query_x[task_idx], adapted_params)
        query_loss = F.mse_loss(query_pred, task_batch.query_y[task_idx])
        meta_loss = meta_loss + query_loss
        inner_loss_sum += inner_loss.detach().item()
        support_loss_sum += support_loss.detach().item()
        ssl_loss_sum += ssl_loss.detach().item()

    meta_loss = meta_loss / task_count
    meta_loss.backward()
    grad_norm = total_grad_norm(adapter.parameters())
    optimizer.step()
    if isinstance(adapter, MetaSGDAdapter):
        adapter.clamp_meta_lrs(min_meta_lr, clamp_meta_lr)

    metrics = {
        "inner_loss": inner_loss_sum / task_count,
        "support_loss": support_loss_sum / task_count,
        "ssl_loss": ssl_loss_sum / task_count,
        "meta_loss": meta_loss.detach().item(),
        "grad_norm": grad_norm,
    }
    metrics.update(adapter.extra_metrics())
    return metrics


def evaluate(
    adapter: MAMLAdapter | MetaSGDAdapter,
    sampler: RegionalTaskSampler,
    val_tasks: int,
    device: torch.device,
    inner_update: str,
    ssl_weight: float,
    inner_steps: int,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    task_batch = sampler.sample(val_tasks, device)
    mse_norm_sum = 0.0
    mae_raw_sum = 0.0
    mse_raw_sum = 0.0

    for task_idx in range(val_tasks):
        adapted_params = None
        for _ in range(inner_steps):
            adapted_params, _, _, _ = adapter.adapt(
                task_batch.support_x[task_idx],
                task_batch.support_y[task_idx],
                task_batch.ssl_x[task_idx],
                task_batch.ssl_y[task_idx],
                task_batch.ssl_confidence[task_idx],
                inner_update=inner_update,
                ssl_weight=ssl_weight,
                params=adapted_params,
                create_graph=False,
            )

        query_pred = adapter.model(task_batch.query_x[task_idx], adapted_params)
        query_y = task_batch.query_y[task_idx]
        mse_norm_sum += F.mse_loss(query_pred, query_y).detach().item()

        pred_raw = query_pred.detach().cpu().numpy() * target_std + target_mean
        y_raw = query_y.detach().cpu().numpy() * target_std + target_mean
        error = pred_raw - y_raw
        mae_raw_sum += float(np.mean(np.abs(error)))
        mse_raw_sum += float(np.mean(error * error))

    mse_norm = mse_norm_sum / val_tasks
    mse_raw = mse_raw_sum / val_tasks
    return {
        "eval_mse_norm": mse_norm,
        "eval_rmse_norm": math.sqrt(max(mse_norm, 0.0)),
        "eval_mse_raw": mse_raw,
        "eval_rmse_raw": math.sqrt(max(mse_raw, 0.0)),
        "eval_mae_raw": mae_raw_sum / val_tasks,
    }


def run_algorithm(
    args: argparse.Namespace,
    dataset: RegionDataset,
    algorithm: str,
) -> dict[str, float]:
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
        algorithm=algorithm,
        input_dim=dataset.features.shape[1],
        hidden_dim=args.hidden_dim,
        inner_lr=args.inner_lr,
        device=device,
    )
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.outer_lr)

    print(
        f"\n[{algorithm}] dataset={dataset.name}, device={device}, "
        f"train_regions={len(dataset.train_regions)}, eval_regions={len(dataset.eval_regions)}, "
        f"features={len(dataset.feature_names)}, target={dataset.target_name}"
    )

    best_eval_mse_norm = float("inf")
    started_at = time.perf_counter()
    final_row: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        task_batch = train_sampler.sample(args.meta_batch, device)
        train_metrics = meta_train_step(
            adapter=adapter,
            optimizer=optimizer,
            task_batch=task_batch,
            first_order=args.first_order,
            inner_update=args.inner_update,
            ssl_weight=args.ssl_weight,
            inner_steps=args.inner_steps,
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
                inner_update=args.inner_update,
                ssl_weight=args.ssl_weight,
                inner_steps=args.inner_steps,
                target_mean=dataset.target_mean,
                target_std=dataset.target_std,
            )
            best_eval_mse_norm = min(best_eval_mse_norm, eval_metrics["eval_mse_norm"])
            elapsed_sec = time.perf_counter() - started_at
            metric_text = " ".join(
                f"{key}={value:.6f}"
                for key, value in {**train_metrics, **eval_metrics}.items()
            )
            print(
                f"step={step:04d} {metric_text} "
                f"best_eval_mse_norm={best_eval_mse_norm:.6f} "
                f"elapsed_sec={elapsed_sec:.2f}"
            )
            final_row = {
                "dataset": dataset.name,
                "algorithm": algorithm,
                "step": step,
                "train_regions": len(dataset.train_regions),
                "eval_regions": len(dataset.eval_regions),
                "feature_count": len(dataset.feature_names),
                "target": dataset.target_name,
                "inner_update": args.inner_update,
                "inner_steps": args.inner_steps,
                "first_order": args.first_order,
                "meta_batch": args.meta_batch,
                "shots": args.shots,
                "queries": args.queries,
                "unlabeled_per_task": args.unlabeled_per_task,
                "ssl_weight": args.ssl_weight,
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


def write_results(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "dataset",
        "algorithm",
        "step",
        "train_regions",
        "eval_regions",
        "feature_count",
        "target",
        "inner_update",
        "inner_steps",
        "first_order",
        "meta_batch",
        "shots",
        "queries",
        "unlabeled_per_task",
        "ssl_weight",
        "outer_lr",
        "inner_lr",
        "inner_loss",
        "support_loss",
        "ssl_loss",
        "meta_loss",
        "eval_mse_norm",
        "eval_rmse_norm",
        "eval_mse_raw",
        "eval_rmse_raw",
        "eval_mae_raw",
        "best_eval_mse_norm",
        "grad_norm",
        "avg_step_time_sec",
        "elapsed_sec",
        "meta_lr_mean",
        "meta_lr_min",
        "meta_lr_max",
        "meta_lr_nonpositive_count",
        "cuda_peak_memory_mb",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-region MAML / Meta-SGD experiments on public datasets.",
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

    parser.add_argument("--algorithms", default="maml,meta-sgd")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--unlabeled-per-task", type=int, default=16)
    parser.add_argument("--eval-tasks", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--outer-lr", type=float, default=1e-3)
    parser.add_argument(
        "--inner-update",
        choices=["supervised", "semi-supervised"],
        default="semi-supervised",
    )
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--min-meta-lr", type=float, default=1e-6)
    parser.add_argument("--clamp-meta-lr", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="public_region_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = make_region_dataset(args)
    algorithms = parse_list(args.algorithms)
    allowed = {"maml", "meta-sgd"}
    for algorithm in algorithms:
        if algorithm not in allowed:
            raise ValueError(f"unknown algorithm: {algorithm}")

    print(
        f"loaded dataset={dataset.name}, rows={len(dataset.targets)}, "
        f"features={dataset.feature_names}, target={dataset.target_name}"
    )
    print(
        f"train_regions={dataset.train_regions}\n"
        f"eval_regions={dataset.eval_regions}"
    )

    rows = [run_algorithm(args, dataset, algorithm) for algorithm in algorithms]
    output_path = Path(args.output)
    write_results(output_path, rows)

    print("\nsummary")
    print(
        "algorithm,eval_mse_norm,eval_rmse_norm,eval_rmse_raw,"
        "eval_mae_raw,best_eval_mse_norm,meta_loss,avg_step_time_sec"
    )
    for row in rows:
        print(
            f"{row['algorithm']},"
            f"{row['eval_mse_norm']:.6f},"
            f"{row['eval_rmse_norm']:.6f},"
            f"{row['eval_rmse_raw']:.6f},"
            f"{row['eval_mae_raw']:.6f},"
            f"{row['best_eval_mse_norm']:.6f},"
            f"{row['meta_loss']:.6f},"
            f"{row['avg_step_time_sec']:.4f}"
        )
    print(f"\nresults saved to: {output_path}")


if __name__ == "__main__":
    main()
