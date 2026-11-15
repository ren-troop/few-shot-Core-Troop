"""The four meta-learning algorithms, written with explicit autograd.

All methods share the same functional inner-loop implementation:

    theta' <- theta' - alpha * grad_theta' L_support(theta')

The differences are which quantities are learned by the outer optimizer:

===========  ==========================================================
MAML         theta (initialization); alpha is a fixed scalar
Meta-SGD     theta and a per-parameter alpha
Fixed-2      a supervised pretrained backbone; only the classifier is
             adapted for two fixed steps (no meta-training)
Meta-L2      theta and a scalar L2 anchor coefficient lambda
===========  ==========================================================

``torch.func.functional_call`` lets us swap the parameter dictionary without
copying the model, which keeps the second-order graph compact and readable.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call

from .data import Episode


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return (logits.argmax(dim=1) == targets).float().mean().item()


class MetaAlgorithm:
    """Base class for episode-level algorithms."""

    name = "base"

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.inner_steps = int(cfg.algorithm.get("inner_steps", 5))
        self.first_order = bool(cfg.algorithm.get("first_order", False))
        self.inner_lr = float(cfg.algorithm.get("inner_lr", 0.01))

    @property
    def parameter_names(self) -> list[str]:
        return [name for name, _ in self.model.named_parameters()]

    def params_dict(self) -> dict[str, torch.Tensor]:
        return {name: param for name, param in self.model.named_parameters()}

    def functional_logits(
        self, params: dict[str, torch.Tensor], images: torch.Tensor
    ) -> torch.Tensor:
        return functional_call(self.model, params, (images,))

    def _grad(
        self,
        loss: torch.Tensor,
        tensors: list[torch.Tensor],
        create_graph: bool,
    ) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            loss,
            tensors,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )

    def _adapt(
        self,
        params: dict[str, torch.Tensor],
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        lr_fn: Callable[[str, int], torch.Tensor | float],
        l2_lambda: torch.Tensor | float | None = None,
        first_order: bool | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[float]]:
        """Run the inner loop and return adapted parameters and step losses."""
        create_graph = not (self.first_order if first_order is None else first_order)
        anchor = {name: value.detach() for name, value in params.items()}
        losses: list[float] = []

        for step in range(self.inner_steps):
            logits = self.functional_logits(params, support_x)
            loss = F.cross_entropy(logits, support_y)
            if l2_lambda is not None:
                penalty = sum(
                    ((params[name] - anchor[name]) ** 2).sum()
                    for name in params
                )
                loss = loss + 0.5 * l2_lambda * penalty
            losses.append(float(loss.detach().cpu()))
            names = list(params.keys())
            grads = self._grad(loss, [params[name] for name in names], create_graph)
            params = {
                name: params[name] - lr_fn(name, step) * grad
                for name, grad in zip(names, grads)
            }
        return params, losses

    def _evaluate_adapted(
        self,
        adapted: dict[str, torch.Tensor],
        episode: Episode,
    ) -> dict[str, float]:
        with torch.no_grad():
            support_logits = self.functional_logits(adapted, episode.support_x)
            query_logits = self.functional_logits(adapted, episode.query_x)
            support_acc = accuracy(support_logits, episode.support_y)
            query_acc = accuracy(query_logits, episode.query_y)
            support_loss = F.cross_entropy(support_logits, episode.support_y).item()
            query_loss = F.cross_entropy(query_logits, episode.query_y).item()
        return {
            "support_acc": support_acc,
            "support_loss": support_loss,
            "query_acc": query_acc,
            "query_loss": query_loss,
        }

    def meta_loss(self, episode: Episode) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def evaluate_episode(self, episode: Episode) -> dict[str, float]:
        raise NotImplementedError

    def extra_parameters(self) -> list[nn.Parameter]:
        return []

    def extra_state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_extra_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            raise KeyError(f"{self.name} has no extra state, got {list(state)}")


class MAML(MetaAlgorithm):
    name = "maml"

    def _adapted(self, episode: Episode, first_order: bool):
        params = self.params_dict()
        return self._adapt(
            params,
            episode.support_x,
            episode.support_y,
            lr_fn=lambda _name, _step: self.inner_lr,
            first_order=first_order,
        )

    def meta_loss(self, episode: Episode) -> tuple[torch.Tensor, dict[str, float]]:
        adapted, inner_losses = self._adapted(episode, self.first_order)
        query_logits = self.functional_logits(adapted, episode.query_x)
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        with torch.no_grad():
            support_acc = accuracy(
                self.functional_logits(adapted, episode.support_x),
                episode.support_y,
            )
        metrics = {
            "query_loss": float(query_loss.detach().cpu()),
            "query_acc": accuracy(query_logits.detach(), episode.query_y),
            "support_acc": support_acc,
            "inner_loss": inner_losses[-1] if inner_losses else 0.0,
        }
        return query_loss, metrics

    def evaluate_episode(self, episode: Episode) -> dict[str, float]:
        adapted, inner_losses = self._adapted(episode, first_order=True)
        metrics = self._evaluate_adapted(adapted, episode)
        metrics["inner_loss"] = inner_losses[-1] if inner_losses else 0.0
        return metrics


class MetaSGD(MetaAlgorithm):
    name = "metasgd"

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        super().__init__(model, cfg, device)
        init_lr = float(cfg.algorithm.get("inner_lr", 0.01))
        if init_lr <= 0:
            raise ValueError("metasgd inner_lr must be positive")
        self.max_lr = float(cfg.algorithm.get("max_inner_lr", 1.0))
        self._names = [name for name, _ in model.named_parameters()]
        self._shapes = [param.shape for _, param in model.named_parameters()]
        self._sizes = [param.numel() for _, param in model.named_parameters()]
        self._offsets: list[tuple[int, int]] = []
        offset = 0
        for size in self._sizes:
            self._offsets.append((offset, offset + size))
            offset += size
        self._index = {name: i for i, name in enumerate(self._names)}
        # inverse softplus so that softplus(log_lr) == init_lr at step 0
        init_log = math.log(math.expm1(init_lr))
        param_device = next(model.parameters()).device
        self.log_lrs = nn.Parameter(
            torch.full((offset,), init_log, device=param_device)
        )

    def lr_for(self, name: str) -> torch.Tensor:
        start, end = self._offsets[self._index[name]]
        lr = F.softplus(self.log_lrs[start:end]).clamp(max=self.max_lr)
        return lr.view(self._shapes[self._index[name]])

    def _adapted(self, episode: Episode, first_order: bool):
        params = self.params_dict()
        return self._adapt(
            params,
            episode.support_x,
            episode.support_y,
            lr_fn=lambda name, _step: self.lr_for(name),
            first_order=first_order,
        )

    def meta_loss(self, episode: Episode) -> tuple[torch.Tensor, dict[str, float]]:
        adapted, inner_losses = self._adapted(episode, self.first_order)
        query_logits = self.functional_logits(adapted, episode.query_x)
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        with torch.no_grad():
            support_acc = accuracy(
                self.functional_logits(adapted, episode.support_x),
                episode.support_y,
            )
            mean_lr = float(F.softplus(self.log_lrs).mean().detach().cpu())
        metrics = {
            "query_loss": float(query_loss.detach().cpu()),
            "query_acc": accuracy(query_logits.detach(), episode.query_y),
            "support_acc": support_acc,
            "inner_loss": inner_losses[-1] if inner_losses else 0.0,
            "mean_lr": mean_lr,
        }
        return query_loss, metrics

    def evaluate_episode(self, episode: Episode) -> dict[str, float]:
        adapted, inner_losses = self._adapted(episode, first_order=True)
        metrics = self._evaluate_adapted(adapted, episode)
        metrics["inner_loss"] = inner_losses[-1] if inner_losses else 0.0
        metrics["mean_lr"] = float(F.softplus(self.log_lrs).mean().detach().cpu())
        return metrics

    def extra_parameters(self) -> list[nn.Parameter]:
        return [self.log_lrs]

    def extra_state_dict(self) -> dict[str, torch.Tensor]:
        return {"log_lrs": self.log_lrs.detach().cpu()}

    def load_extra_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "log_lrs" in state:
            self.log_lrs.data.copy_(state["log_lrs"].to(self.log_lrs.device))


class MetaL2(MetaAlgorithm):
    name = "metal2"

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        super().__init__(model, cfg, device)
        init_lambda = float(cfg.algorithm.get("l2_lambda", 0.001))
        if init_lambda <= 0:
            raise ValueError("metal2 l2_lambda must be positive")
        self.learn_lambda = bool(cfg.algorithm.get("learn_l2_lambda", True))
        param_device = next(model.parameters()).device
        self.log_lambda = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(init_lambda)),
                dtype=torch.float32,
                device=param_device,
            )
        )
        if not self.learn_lambda:
            self.log_lambda.requires_grad_(False)

    @property
    def l2_lambda(self) -> torch.Tensor:
        return F.softplus(self.log_lambda)

    def _adapted(self, episode: Episode, first_order: bool):
        params = self.params_dict()
        return self._adapt(
            params,
            episode.support_x,
            episode.support_y,
            lr_fn=lambda _name, _step: self.inner_lr,
            l2_lambda=self.l2_lambda,
            first_order=first_order,
        )

    def meta_loss(self, episode: Episode) -> tuple[torch.Tensor, dict[str, float]]:
        adapted, inner_losses = self._adapted(episode, self.first_order)
        query_logits = self.functional_logits(adapted, episode.query_x)
        query_loss = F.cross_entropy(query_logits, episode.query_y)
        with torch.no_grad():
            support_acc = accuracy(
                self.functional_logits(adapted, episode.support_x),
                episode.support_y,
            )
        metrics = {
            "query_loss": float(query_loss.detach().cpu()),
            "query_acc": accuracy(query_logits.detach(), episode.query_y),
            "support_acc": support_acc,
            "inner_loss": inner_losses[-1] if inner_losses else 0.0,
            "l2_lambda": float(self.l2_lambda.detach().cpu()),
        }
        return query_loss, metrics

    def evaluate_episode(self, episode: Episode) -> dict[str, float]:
        adapted, inner_losses = self._adapted(episode, first_order=True)
        metrics = self._evaluate_adapted(adapted, episode)
        metrics["inner_loss"] = inner_losses[-1] if inner_losses else 0.0
        metrics["l2_lambda"] = float(self.l2_lambda.detach().cpu())
        return metrics

    def extra_parameters(self) -> list[nn.Parameter]:
        return [self.log_lambda]

    def extra_state_dict(self) -> dict[str, torch.Tensor]:
        return {"log_lambda": self.log_lambda.detach().cpu()}

    def load_extra_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if "log_lambda" in state:
            self.log_lambda.data.copy_(state["log_lambda"].to(self.log_lambda.device))


class Fixed2(MetaAlgorithm):
    """Pretrained, frozen backbone + two gradient steps on a new classifier.

    This is the "fixed-2" baseline: the representation is fixed by ordinary
    supervised pretraining on the training classes, and each episode adapts a
    fresh linear head for exactly two steps.  Set ``freeze_backbone: false`` to
    fine-tune the whole network instead (a different, stronger baseline).
    """

    name = "fixed2"

    def __init__(self, model: nn.Module, cfg, device: torch.device):
        super().__init__(model, cfg, device)
        self.inner_steps = int(cfg.algorithm.get("inner_steps", 2))
        self.freeze_backbone = bool(cfg.algorithm.get("freeze_backbone", True))
        self.n_way = int(cfg.dataset.n_way)
        self._adapt_names: list[str] = []

    def prepare_for_classes(self, num_classes: int) -> None:
        self.model.reset_classifier(int(num_classes))
        if self.freeze_backbone:
            self._adapt_names = [
                name
                for name, _ in self.model.named_parameters()
                if name.startswith("classifier.")
            ]
        else:
            self._adapt_names = [name for name, _ in self.model.named_parameters()]
        if not self._adapt_names:
            raise RuntimeError("Fixed2 has no parameters to adapt")

    def _adapted(self, episode: Episode):
        params = self.params_dict()
        if self.freeze_backbone:
            # Detach the backbone so its activations do not build a graph.
            params = {
                name: (value if name in self._adapt_names else value.detach())
                for name, value in params.items()
            }
        for _step in range(self.inner_steps):
            logits = self.functional_logits(params, episode.support_x)
            loss = F.cross_entropy(logits, episode.support_y)
            grads = self._grad(
                loss,
                [params[name] for name in self._adapt_names],
                create_graph=False,
            )
            for name, grad in zip(self._adapt_names, grads):
                params[name] = params[name] - self.inner_lr * grad
        return params

    def meta_loss(self, episode: Episode) -> tuple[torch.Tensor, dict[str, float]]:
        raise RuntimeError("Fixed2 is trained by supervised pretraining, not meta_loss")

    def evaluate_episode(self, episode: Episode) -> dict[str, float]:
        # A fresh head is required for every episode; otherwise the previous
        # episode's classifier would leak into the next one.
        self.prepare_for_classes(self.n_way)
        adapted = self._adapted(episode)
        metrics = self._evaluate_adapted(adapted, episode)
        metrics["inner_loss"] = metrics["support_loss"]
        metrics["l2_lambda"] = 0.0
        return metrics

    def extra_parameters(self) -> list[nn.Parameter]:
        return []


def build_algorithm(cfg, model: nn.Module, device: torch.device) -> MetaAlgorithm:
    name = str(cfg.algorithm.name).lower().replace("_", "-")
    mapping = {
        "maml": MAML,
        "metasgd": MetaSGD,
        "meta-sgd": MetaSGD,
        "metal2": MetaL2,
        "meta-l2": MetaL2,
        "fixed2": Fixed2,
        "fixed-2": Fixed2,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown algorithm {cfg.algorithm.name!r}; "
            f"choose from {sorted(mapping)}"
        )
    return mapping[name](model, cfg, device)


def optimizer_parameter_groups(
    model: nn.Module,
    algorithm: MetaAlgorithm,
    meta_lr: float,
    hyper_lr: float,
    weight_decay: float,
) -> list[dict]:
    groups: list[dict] = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": float(meta_lr),
            "weight_decay": float(weight_decay),
        }
    ]
    extra = [p for p in algorithm.extra_parameters() if p.requires_grad]
    if extra:
        groups.append(
            {"params": extra, "lr": float(hyper_lr), "weight_decay": 0.0}
        )
    return groups
