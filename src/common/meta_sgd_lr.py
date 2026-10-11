"""Shared Meta-SGD inner learning-rate constraints and reporting."""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_CLAMP_MIN = 1e-6
DEFAULT_CLAMP_MAX = 0.2
MODES = ("softplus", "clamp", "none")


def _validate_bounds(clamp_min: float, clamp_max: float) -> None:
    if (not math.isfinite(clamp_min) or not math.isfinite(clamp_max)
            or clamp_min <= 0 or clamp_max < clamp_min):
        raise ValueError("clamp bounds must be finite, positive, and clamp_max >= clamp_min")


def apply_meta_sgd_constraint(
    raw: Tensor,
    mode: str = "softplus",
    clamp_min: float = DEFAULT_CLAMP_MIN,
    clamp_max: float = DEFAULT_CLAMP_MAX,
) -> Tensor:
    """Map raw learned steps to softplus, clamp, or unchanged effective steps."""
    if mode not in MODES:
        raise ValueError(f"Unknown Meta-SGD mode {mode!r}; expected one of {MODES}")
    _validate_bounds(clamp_min, clamp_max)
    if mode == "softplus":
        return F.softplus(raw)
    if mode == "clamp":
        return raw.clamp(min=clamp_min, max=clamp_max)
    return raw


def summarize_lr(
    values: Iterable[Tensor],
    *,
    raw_values: Iterable[Tensor] | None = None,
    mode: str = "none",
    clamp_min: float = DEFAULT_CLAMP_MIN,
    clamp_max: float = DEFAULT_CLAMP_MAX,
) -> dict[str, float | int]:
    """Summarize effective steps; count values changed by clamp when raw values are supplied."""
    if mode not in MODES:
        raise ValueError(f"Unknown Meta-SGD mode {mode!r}; expected one of {MODES}")
    _validate_bounds(clamp_min, clamp_max)
    effective_parts = [v.detach().reshape(-1).to(device="cpu", dtype=torch.float64) for v in values]
    if not effective_parts or sum(part.numel() for part in effective_parts) == 0:
        raise ValueError("At least one nonempty learning-rate tensor is required")
    effective = torch.cat(effective_parts)
    if not torch.isfinite(effective).all():
        raise ValueError("Learning-rate values must be finite")

    clamp_count = 0
    if raw_values is not None:
        raw_parts = [v.detach().reshape(-1).to(device="cpu", dtype=torch.float64) for v in raw_values]
        if len(raw_parts) != len(effective_parts) or any(
                a.numel() != b.numel() for a, b in zip(raw_parts, effective_parts)):
            raise ValueError("raw_values must match values in length and shape")
        raw = torch.cat(raw_parts)
        if not torch.isfinite(raw).all():
            raise ValueError("Raw learning-rate values must be finite")
        if mode == "clamp":
            clamp_count = int((raw != effective).sum().item())
    elif mode == "clamp":
        raise ValueError("raw_values are required to report clamp_count in clamp mode")

    return {
        "mean": float(effective.mean().item()),
        "std": float(effective.std(unbiased=False).item()),
        "min": float(effective.min().item()),
        "max": float(effective.max().item()),
        "negative_count": int((effective < 0).sum().item()),
        "zero_count": int((effective == 0).sum().item()),
        "positive_count": int((effective > 0).sum().item()),
        "clamp_count": clamp_count,
    }
