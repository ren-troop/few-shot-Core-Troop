from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

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


class SharedBackboneRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 40) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        params: OrderedDict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            features = self.backbone(x)
            return self.head(features)

        x = F.linear(x, params["backbone.0.weight"], params["backbone.0.bias"])
        x = torch.tanh(x)
        x = F.linear(x, params["backbone.2.weight"], params["backbone.2.bias"])
        x = torch.tanh(x)
        x = F.linear(x, params["head.weight"], params["head.bias"])
        return x


class FewShotTaskSampler:
    def __init__(
        self,
        shots: int,
        queries: int,
        unlabeled_per_task: int,
        pseudo_noise: float,
        amplitude_range: tuple[float, float] = (0.1, 5.0),
        phase_range: tuple[float, float] = (0.0, math.pi),
        x_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        self.shots = shots
        self.queries = queries
        self.unlabeled_per_task = unlabeled_per_task
        self.pseudo_noise = pseudo_noise
        self.amplitude_range = amplitude_range
        self.phase_range = phase_range
        self.x_range = x_range

    def sample(self, num_tasks: int, device: torch.device) -> TaskBatch:
        amplitudes = torch.empty(num_tasks, 1, 1, device=device).uniform_(
            *self.amplitude_range,
        )
        phases = torch.empty(num_tasks, 1, 1, device=device).uniform_(*self.phase_range)
        support_x = torch.empty(num_tasks, self.shots, 1, device=device).uniform_(
            *self.x_range,
        )
        query_x = torch.empty(num_tasks, self.queries, 1, device=device).uniform_(
            *self.x_range,
        )
        support_y = amplitudes * torch.sin(support_x + phases)
        query_y = amplitudes * torch.sin(query_x + phases)

        ssl_x = torch.empty(
            num_tasks,
            self.unlabeled_per_task,
            1,
            device=device,
        ).uniform_(*self.x_range)
        ssl_true_y = amplitudes * torch.sin(ssl_x + phases)
        pseudo_noise = self.pseudo_noise * torch.randn_like(ssl_true_y)
        ssl_y = ssl_true_y + pseudo_noise
        ssl_confidence = torch.exp(-pseudo_noise.abs()).detach()

        return TaskBatch(
            support_x,
            support_y,
            query_x,
            query_y,
            ssl_x,
            ssl_y,
            ssl_confidence,
        )


def compute_inner_loss(
    model: SharedBackboneRegressor,
    params: OrderedDict[str, torch.Tensor],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    ssl_x: torch.Tensor,
    ssl_y: torch.Tensor,
    ssl_confidence: torch.Tensor,
    inner_update: str,
    ssl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_pred = model(support_x, params)
    supervised_loss = F.mse_loss(support_pred, support_y)

    if inner_update != "semi-supervised" or ssl_x.numel() == 0 or ssl_weight <= 0:
        ssl_loss = supervised_loss.new_zeros(())
        return supervised_loss, supervised_loss, ssl_loss

    ssl_pred = model(ssl_x, params)
    ssl_loss_per_sample = F.mse_loss(ssl_pred, ssl_y, reduction="none")
    ssl_loss = (ssl_loss_per_sample * ssl_confidence).sum() / ssl_confidence.sum().clamp_min(1.0)
    total_loss = supervised_loss + ssl_weight * ssl_loss
    return total_loss, supervised_loss, ssl_loss


class MAMLAdapter(nn.Module):
    def __init__(self, model: SharedBackboneRegressor, inner_lr: float) -> None:
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
        inner_loss, supervised_loss, ssl_loss = compute_inner_loss(
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
        return adapted_params, inner_loss, supervised_loss, ssl_loss

    def extra_metrics(self) -> dict[str, float]:
        return {"inner_lr": self.inner_lr}


class MetaSGDAdapter(nn.Module):
    def __init__(self, model: SharedBackboneRegressor, init_inner_lr: float) -> None:
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
        inner_loss, supervised_loss, ssl_loss = compute_inner_loss(
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
        return adapted_params, inner_loss, supervised_loss, ssl_loss

    def extra_metrics(self) -> dict[str, float]:
        flat_lrs = torch.cat([meta_lr.detach().flatten() for meta_lr in self.meta_lrs])
        return {
            "meta_lr_mean": flat_lrs.mean().item(),
            "meta_lr_min": flat_lrs.min().item(),
            "meta_lr_max": flat_lrs.max().item(),
            "meta_lr_nonpositive_count": (flat_lrs <= 0).sum().item(),
        }

    def clamp_meta_lrs(self, min_value: float, max_value: float) -> None:
        if max_value <= 0:
            return
        lower = max(min_value, 1e-12)
        upper = max(max_value, lower)
        with torch.no_grad():
            for meta_lr in self.meta_lrs:
                meta_lr.clamp_(min=lower, max=upper)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def total_grad_norm(parameters: Iterable[torch.Tensor]) -> float:
    squared_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            squared_norm += param.grad.detach().pow(2).sum().item()
    return squared_norm ** 0.5


def build_adapter(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = SharedBackboneRegressor(hidden_dim=args.hidden_dim).to(device)
    if args.algorithm == "maml":
        return MAMLAdapter(model, inner_lr=args.inner_lr).to(device)
    if args.algorithm == "meta-sgd":
        return MetaSGDAdapter(model, init_inner_lr=args.inner_lr).to(device)
    raise ValueError(f"unknown algorithm: {args.algorithm}")


def meta_train_step(
    adapter: MAMLAdapter | MetaSGDAdapter,
    optimizer: torch.optim.Optimizer,
    task_batch: TaskBatch,
    first_order: bool,
    clamp_meta_lr: float,
    min_meta_lr: float,
    inner_update: str,
    ssl_weight: float,
    inner_steps: int,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    num_tasks = task_batch.support_x.shape[0]
    meta_loss = torch.zeros((), device=task_batch.support_x.device)
    inner_loss_sum = 0.0
    support_loss_sum = 0.0
    ssl_loss_sum = 0.0

    for task_idx in range(num_tasks):
        adapted_params = None
        inner_loss = torch.zeros((), device=task_batch.support_x.device)
        supervised_loss = torch.zeros((), device=task_batch.support_x.device)
        ssl_loss = torch.zeros((), device=task_batch.support_x.device)

        for _ in range(inner_steps):
            adapted_params, inner_loss, supervised_loss, ssl_loss = adapter.adapt(
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
        support_loss_sum += supervised_loss.detach().item()
        ssl_loss_sum += ssl_loss.detach().item()

    meta_loss = meta_loss / num_tasks
    meta_loss.backward()
    grad_norm = total_grad_norm(adapter.parameters())
    optimizer.step()

    if isinstance(adapter, MetaSGDAdapter):
        adapter.clamp_meta_lrs(min_meta_lr, clamp_meta_lr)

    metrics = {
        "inner_loss": inner_loss_sum / num_tasks,
        "support_loss": support_loss_sum / num_tasks,
        "ssl_loss": ssl_loss_sum / num_tasks,
        "meta_loss": meta_loss.detach().item(),
        "grad_norm": grad_norm,
    }
    metrics.update(adapter.extra_metrics())
    return metrics


def evaluate(
    adapter: MAMLAdapter | MetaSGDAdapter,
    sampler: FewShotTaskSampler,
    val_tasks: int,
    device: torch.device,
    inner_update: str,
    ssl_weight: float,
    inner_steps: int,
) -> float:
    task_batch = sampler.sample(val_tasks, device)
    query_loss_sum = 0.0

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
        query_loss_sum += F.mse_loss(
            query_pred,
            task_batch.query_y[task_idx],
        ).detach().item()

    return query_loss_sum / val_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified framework for MAML and Meta-SGD few-shot regression.",
    )
    parser.add_argument("--algorithm", choices=["maml", "meta-sgd"], default="meta-sgd")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--meta-batch", type=int, default=8)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--unlabeled-per-task", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--inner-steps", type=int, default=1)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        f"algorithm={args.algorithm}, device={device}, "
        f"inner_update={args.inner_update}, inner_steps={args.inner_steps}, "
        f"first_order={args.first_order}, meta_batch={args.meta_batch}"
    )
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
            inner_steps=args.inner_steps,
        )

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val_query_loss = evaluate(
                adapter,
                sampler,
                args.val_tasks,
                device,
                inner_update=args.inner_update,
                ssl_weight=args.ssl_weight,
                inner_steps=args.inner_steps,
            )
            metric_text = " ".join(
                f"{key}={value:.6f}" for key, value in metrics.items()
            )
            print(
                f"step={step:04d} {metric_text} "
                f"val_query_loss={val_query_loss:.6f}"
            )


if __name__ == "__main__":
    main()
