"""Episodic datasets and the N-way K-shot episode sampler.

Expected on-disk layout after running ``scripts/download_data.py``::

    data/
      cifar-fs/
        train/<class_name>/*.png
        val/<class_name>/*.png
        test/<class_name>/*.png

Each split directory contains only the classes that belong to that split.  The
sampler never sees class names, so labels are local to each episode (0..N-1).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass
class Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    class_names: list[str]

    def to(self, device: torch.device) -> "Episode":
        return Episode(
            support_x=self.support_x.to(device, non_blocking=True),
            support_y=self.support_y.to(device, non_blocking=True),
            query_x=self.query_x.to(device, non_blocking=True),
            query_y=self.query_y.to(device, non_blocking=True),
            class_names=list(self.class_names),
        )


def build_transform(
    image_size: int,
    train: bool,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    ops: list = [
        transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BILINEAR
        )
    ]
    if train:
        ops.extend(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        ops.append(transforms.CenterCrop(image_size))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(tuple(mean), tuple(std)),
        ]
    )
    return transforms.Compose(ops)


class FolderDataset(Dataset):
    """A split stored as ``<split>/<class>/*.jpg|png``.

    Parameters
    ----------
    root:
        Directory for one split, e.g. ``data/cifar-fs/train``.
    transform:
        Torchvision transform applied to every PIL image.
    cache_images:
        Keep decoded uint8 arrays in RAM.  Useful for CIFAR-FS (about 180 MB);
        leave off for larger 84x84 datasets on a 16 GB machine.
    """

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path,
        transform=None,
        cache_images: bool = False,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory not found: {self.root}\n"
                "Run: python scripts/download_data.py"
            )
        self.transform = transform
        self.classes = sorted(
            [p.name for p in self.root.iterdir() if p.is_dir()],
            key=lambda name: name.lower(),
        )
        if not self.classes:
            raise RuntimeError(f"No class folders found under {self.root}")

        self.paths_by_class: list[list[Path]] = []
        for class_name in self.classes:
            paths = sorted(
                p
                for p in (self.root / class_name).iterdir()
                if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES
            )
            if not paths:
                raise RuntimeError(f"Class {class_name!r} has no images in {self.root}")
            self.paths_by_class.append(paths)

        self._cumulative: list[int] = []
        total = 0
        for paths in self.paths_by_class:
            total += len(paths)
            self._cumulative.append(total)

        self._cache: list[list[np.ndarray]] | None = None
        if cache_images:
            self._cache = []
            for paths in self.paths_by_class:
                self._cache.append([self._read_uint8(p) for p in paths])

    @staticmethod
    def _read_uint8(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def _load(self, class_idx: int, image_idx: int) -> Image.Image:
        if self._cache is not None:
            return Image.fromarray(self._cache[class_idx][image_idx], mode="RGB")
        with Image.open(self.paths_by_class[class_idx][image_idx]) as image:
            return image.convert("RGB")

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, index: int):
        # Index over the concatenation of all classes, used by the pretraining
        # DataLoader rather than by episode sampling.
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        class_idx = bisect.bisect_right(self._cumulative, index)
        previous = self._cumulative[class_idx - 1] if class_idx > 0 else 0
        image = self._load(class_idx, index - previous)
        if self.transform is not None:
            image = self.transform(image)
        return image, class_idx

    def flat_items(self) -> list[tuple[Path, int]]:
        items: list[tuple[Path, int]] = []
        for class_idx, paths in enumerate(self.paths_by_class):
            items.extend((path, class_idx) for path in paths)
        return items

    def sample_episode(
        self,
        rng: np.random.Generator,
        n_way: int,
        n_shot: int,
        n_query: int,
    ) -> Episode:
        if n_way > len(self.classes):
            raise ValueError(
                f"Cannot sample {n_way}-way from {len(self.classes)} classes in {self.root}"
            )
        class_indices = rng.choice(len(self.classes), size=n_way, replace=False)
        support_images: list[torch.Tensor] = []
        query_images: list[torch.Tensor] = []
        support_labels: list[int] = []
        query_labels: list[int] = []
        class_names: list[str] = []

        for local_label, class_idx in enumerate(class_indices):
            paths = self.paths_by_class[int(class_idx)]
            needed = n_shot + n_query
            if len(paths) < needed:
                raise ValueError(
                    f"Class {self.classes[int(class_idx)]!r} has {len(paths)} images, "
                    f"but n_shot+n_query={needed}."
                )
            chosen = rng.choice(len(paths), size=needed, replace=False)
            class_names.append(self.classes[int(class_idx)])
            for position, image_idx in enumerate(chosen):
                image = self._load(int(class_idx), int(image_idx))
                if self.transform is not None:
                    image = self.transform(image)
                if position < n_shot:
                    support_images.append(image)
                    support_labels.append(local_label)
                else:
                    query_images.append(image)
                    query_labels.append(local_label)

        return Episode(
            support_x=torch.stack(support_images, dim=0),
            support_y=torch.as_tensor(support_labels, dtype=torch.long),
            query_x=torch.stack(query_images, dim=0),
            query_y=torch.as_tensor(query_labels, dtype=torch.long),
            class_names=class_names,
        )


class EpisodeGenerator:
    """Infinite-ish generator of reproducible episodes for one split."""

    def __init__(
        self,
        dataset: FolderDataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.n_way = int(n_way)
        self.n_shot = int(n_shot)
        self.n_query = int(n_query)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> Episode:
        return self.dataset.sample_episode(
            self.rng, self.n_way, self.n_shot, self.n_query
        )

    def batches(self, batch_size: int) -> Iterator[list[Episode]]:
        while True:
            yield [self.sample() for _ in range(int(batch_size))]


def dataset_root(cfg, split: str) -> Path:
    return Path(cfg.dataset.root) / str(cfg.dataset.name) / split


def build_dataset(cfg, split: str, train: bool) -> FolderDataset:
    transform = build_transform(
        image_size=int(cfg.dataset.image_size),
        train=train,
        mean=cfg.dataset.get("mean", (0.485, 0.456, 0.406)),
        std=cfg.dataset.get("std", (0.229, 0.224, 0.225)),
    )
    return FolderDataset(
        dataset_root(cfg, split),
        transform=transform,
        cache_images=bool(cfg.dataset.get("cache_images", False)),
    )


def build_generator(cfg, split: str, train: bool, seed: int) -> EpisodeGenerator:
    dataset = build_dataset(cfg, split, train=train)
    return EpisodeGenerator(
        dataset,
        n_way=int(cfg.dataset.n_way),
        n_shot=int(cfg.dataset.n_shot),
        n_query=int(cfg.dataset.n_query),
        seed=seed,
    )


def split_summary(root: str | Path) -> dict:
    """Return class/image counts, used by the downloader and tests."""
    root = Path(root)
    if not root.is_dir():
        return {"exists": False, "classes": 0, "images": 0}
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    images = 0
    for class_dir in class_dirs:
        images += sum(
            1
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in FolderDataset.IMAGE_SUFFIXES
        )
    return {"exists": True, "classes": len(class_dirs), "images": images}
