from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call

from .data import Episode


METHODS = ("maml", "meta_sgd", "fixed_l2", "meta_l2")


def _safe_key(parameter_name: str) -> str:
    return parameter_name.replace(".", "__")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MetaLearner(nn.Module):
    """MAML、Meta-SGD 风格学习率元学习及可学习 L2 正则化。"""

    def __init__(
        self,
        model: nn.Module,
        method: str,
        inner_lr: float,
        inner_steps: int,
        initial_l2: float,
        first_order: bool,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"未知方法 {method}，应为 {METHODS} 之一。")
        if method == "meta_l2" and first_order:
            raise ValueError("meta_l2 需要二阶梯度，不能同时使用 --first-order。")

        self.model = model
        self.method = method
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.initial_l2 = float(initial_l2)
        self.first_order = bool(first_order)

        self.learned_lrs = nn.ParameterDict()
        self.raw_l2 = nn.ParameterDict()
        for name, parameter in self.model.named_parameters():
            key = _safe_key(name)
            if method == "meta_sgd":
                self.learned_lrs[key] = nn.Parameter(
                    torch.full_like(parameter, fill_value=inner_lr)
                )
            if method == "meta_l2" and parameter.ndim > 1:
                self.raw_l2[key] = nn.Parameter(
                    torch.tensor(_inverse_softplus(initial_l2), dtype=parameter.dtype)
                )

    def _support_loss(
        self, params: OrderedDict[str, Tensor], images: Tensor, labels: Tensor
    ) -> Tensor:
        logits = functional_call(self.model, params, (images,))
        loss = F.cross_entropy(logits, labels)
        if self.method in ("fixed_l2", "meta_l2"):
            regularization = torch.zeros((), device=loss.device, dtype=loss.dtype)
            for name, parameter in params.items():
                key = _safe_key(name)
                if self.method == "meta_l2" and key in self.raw_l2:
                    coefficient = F.softplus(self.raw_l2[key])
                    regularization = regularization + coefficient * parameter.square().mean()
                elif self.method == "fixed_l2" and parameter.ndim > 1:
                    regularization = (
                        regularization + self.initial_l2 * parameter.square().mean()
                    )
            loss = loss + regularization
        return loss

    def adapt(
        self, support_x: Tensor, support_y: Tensor, training: bool
    ) -> OrderedDict[str, Tensor]:
        params = OrderedDict(self.model.named_parameters())
        create_graph = training and not self.first_order

        for _ in range(self.inner_steps):
            loss = self._support_loss(params, support_x, support_y)
            gradients = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            updated = OrderedDict()
            for (name, parameter), gradient in zip(params.items(), gradients):
                if self.method == "meta_sgd":
                    step_size: Tensor | float = self.learned_lrs[_safe_key(name)]
                else:
                    step_size = self.inner_lr
                updated[name] = parameter - step_size * gradient
            params = updated
        return params

    def episode_metrics(self, episode: Episode, training: bool) -> tuple[Tensor, Tensor]:
        adapted = self.adapt(episode.support_x, episode.support_y, training=training)
        query_logits = functional_call(self.model, adapted, (episode.query_x,))
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        query_accuracy = (query_logits.argmax(dim=1) == episode.query_y).float().mean()
        return query_loss, query_accuracy

    def learned_hyperparameter_rows(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        if self.method == "meta_sgd":
            values: Iterable[tuple[str, Tensor]] = self.learned_lrs.items()
            label = "learning_rate"
        elif self.method == "meta_l2":
            values = ((name, F.softplus(value)) for name, value in self.raw_l2.items())
            label = "l2_coefficient"
        elif self.method == "fixed_l2":
            return [
                {
                    "method": self.method,
                    "kind": "fixed_l2_coefficient",
                    "parameter": "all_weight_layers",
                    "mean": self.initial_l2,
                    "std": 0.0,
                    "min": self.initial_l2,
                    "max": self.initial_l2,
                }
            ]
        else:
            return rows

        for name, value in values:
            detached = value.detach().float().cpu()
            rows.append(
                {
                    "method": self.method,
                    "kind": label,
                    "parameter": name.replace("__", "."),
                    "mean": float(detached.mean()),
                    "std": float(detached.std(unbiased=False)),
                    "min": float(detached.min()),
                    "max": float(detached.max()),
                }
            )
        return rows
