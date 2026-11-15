"""The four learners: MAML, Meta-SGD, Meta-L2, and Fixed-2."""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call
from torch.utils.data import DataLoader

from .data import Episode, OmniglotEpisodeSampler, OmniglotPretrainDataset

METHODS = ("maml", "meta_sgd", "meta_l2", "fixed2")


def _safe_key(name: str) -> str:
    return name.replace(".", "__")


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("initial value must be finite and > 0")
    return value + math.log(-math.expm1(-value))


class MetaLearner(nn.Module):
    """MAML / Meta-SGD / Meta-L2 with functional inner-loop updates."""

    def __init__(
        self,
        model: nn.Module,
        method: str,
        inner_lr: float,
        inner_steps: int,
        initial_l2: float,
        first_order: bool,
        max_inner_lr: float = 1.0,
    ) -> None:
        super().__init__()
        if method not in ("maml", "meta_sgd", "meta_l2"):
            raise ValueError(f"Unknown meta method: {method}")
        if method == "meta_l2" and first_order:
            raise ValueError("Meta-L2 requires second-order gradients")
        self.model = model
        self.method = method
        self.inner_lr = float(inner_lr)
        self.inner_steps = int(inner_steps)
        self.first_order = bool(first_order)
        self.max_inner_lr = float(max_inner_lr)

        self.raw_lrs = nn.ParameterDict()
        if method == "meta_sgd":
            init = _inverse_softplus(self.inner_lr)
            for name, parameter in model.named_parameters():
                self.raw_lrs[_safe_key(name)] = nn.Parameter(
                    torch.full_like(parameter, init)
                )

        self.log_lambda: nn.Parameter | None = None
        if method == "meta_l2":
            self.log_lambda = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(float(initial_l2)),
                    dtype=torch.float32,
                    device=next(model.parameters()).device,
                )
            )

    def _support_loss(
        self,
        params: OrderedDict[str, Tensor],
        images: Tensor,
        labels: Tensor,
        anchor: dict[str, Tensor],
    ) -> Tensor:
        logits = functional_call(self.model, params, (images,))
        loss = F.cross_entropy(logits, labels)
        if self.method == "meta_l2" and self.log_lambda is not None:
            penalty = sum(
                ((params[name] - anchor[name]) ** 2).sum() for name in params
            )
            loss = loss + 0.5 * F.softplus(self.log_lambda) * penalty
        return loss

    def adapt(
        self, support_x: Tensor, support_y: Tensor, training: bool
    ) -> OrderedDict[str, Tensor]:
        params = OrderedDict(self.model.named_parameters())
        anchor = {name: value.detach() for name, value in params.items()}
        create_graph = training and not self.first_order

        for _ in range(self.inner_steps):
            loss = self._support_loss(params, support_x, support_y, anchor)
            gradients = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            updated = OrderedDict()
            for (name, parameter), gradient in zip(params.items(), gradients):
                if self.method == "meta_sgd":
                    step = F.softplus(self.raw_lrs[_safe_key(name)]).clamp(
                        max=self.max_inner_lr
                    )
                else:
                    step = self.inner_lr
                updated[name] = parameter - step * gradient
            params = updated
        return params

    def episode_metrics(self, episode: Episode, training: bool) -> tuple[Tensor, Tensor]:
        adapted = self.adapt(episode.support_x, episode.support_y, training=training)
        logits = functional_call(self.model, adapted, (episode.query_x,))
        loss = F.cross_entropy(logits, episode.query_y)
        accuracy = (logits.argmax(dim=1) == episode.query_y).float().mean()
        return loss, accuracy

    def hyperparameter_summary(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        if self.method == "meta_sgd" and self.raw_lrs:
            values = torch.cat(
                [F.softplus(value).detach().reshape(-1) for value in self.raw_lrs.values()]
            )
            summary["mean_inner_lr"] = float(values.mean().cpu())
            summary["min_inner_lr"] = float(values.min().cpu())
            summary["max_inner_lr"] = float(values.max().cpu())
        if self.method == "meta_l2" and self.log_lambda is not None:
            summary["l2_lambda"] = float(F.softplus(self.log_lambda).detach().cpu())
        return summary

    def l2_parameter(self) -> nn.Parameter | None:
        return self.log_lambda if self.method == "meta_l2" else None


class Fixed2Learner(nn.Module):
    """Pretrained frozen ConvNet plus a fresh 5-way head adapted for two steps."""

    def __init__(
        self,
        model: nn.Module,
        n_way: int,
        inner_steps: int = 2,
        inner_lr: float = 0.05,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_way = int(n_way)
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.freeze_backbone = bool(freeze_backbone)

    def pretrain(
        self,
        sampler: OmniglotEpisodeSampler,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        device: torch.device,
        log_path: Path,
    ) -> None:
        dataset = OmniglotPretrainDataset(sampler)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * len(loader))
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["epoch", "loss", "accuracy", "images"]
            )
            writer.writeheader()
            self.model.train()
            for epoch in range(1, epochs + 1):
                total_loss = 0.0
                correct = 0
                seen = 0
                for images, labels in loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(images)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item() * labels.numel()
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    seen += labels.numel()
                row = {
                    "epoch": epoch,
                    "loss": total_loss / max(1, seen),
                    "accuracy": correct / max(1, seen),
                    "images": seen,
                }
                writer.writerow(row)
                handle.flush()
                print(
                    f"[fixed2 pretrain] epoch={epoch:03d} "
                    f"loss={row['loss']:.4f} acc={row['accuracy']:.4f}",
                    flush=True,
                )
        self.model.eval()

    def _adapted(self, episode: Episode) -> OrderedDict[str, Tensor]:
        self.model.reset_classifier(self.n_way)
        params = OrderedDict(self.model.named_parameters())
        if self.freeze_backbone:
            adapt_names = [name for name in params if name.startswith("classifier.")]
            params = {
                name: (value if name in adapt_names else value.detach())
                for name, value in params.items()
            }
        else:
            adapt_names = list(params)

        for _ in range(self.inner_steps):
            logits = functional_call(self.model, params, (episode.support_x,))
            loss = F.cross_entropy(logits, episode.support_y)
            gradients = torch.autograd.grad(
                loss,
                tuple(params[name] for name in adapt_names),
                create_graph=False,
            )
            params = dict(params)
            for name, gradient in zip(adapt_names, gradients):
                params[name] = params[name] - self.inner_lr * gradient
        return OrderedDict(params)

    def episode_metrics(self, episode: Episode, training: bool = False) -> tuple[Tensor, Tensor]:
        adapted = self._adapted(episode)
        with torch.no_grad():
            logits = functional_call(self.model, adapted, (episode.query_x,))
            loss = F.cross_entropy(logits, episode.query_y)
            accuracy = (logits.argmax(dim=1) == episode.query_y).float().mean()
        return loss, accuracy

    def hyperparameter_summary(self) -> dict[str, float]:
        return {
            "fixed2_inner_steps": float(self.inner_steps),
            "fixed2_inner_lr": self.inner_lr,
        }
