"""Small utilities shared by training, evaluation, and data preparation."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/``cuda``/``cpu`` and fail early for a bad CUDA request."""
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA build of PyTorch and check the NVIDIA driver."
        )
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"CUDA:{index} {name} ({total:.1f} GiB)"
    return f"CPU ({os.cpu_count()} threads)"


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class CSVLogger:
    """Append-only CSV logger that writes the header on first use."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

    def log(self, row: dict[str, Any]) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start


def cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"GPU mem allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB"
