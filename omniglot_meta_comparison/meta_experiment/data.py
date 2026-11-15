"""Omniglot episode sampling and a flat dataset for Fixed-2 pretraining."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass
class Episode:
    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor

    def to(self, device: torch.device) -> "Episode":
        return Episode(
            self.support_x.to(device),
            self.support_y.to(device),
            self.query_x.to(device),
            self.query_y.to(device),
        )


def _transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            # Omniglot stores white strokes on black; images are inverted so
            # the input resembles the usual MNIST-style white background.
            transforms.Lambda(lambda image: 1.0 - image),
        ]
    )


class OmniglotEpisodeSampler:
    """Sample N-way K-shot tasks from a disjoint set of Omniglot characters."""

    def __init__(
        self,
        root: str | Path,
        background: bool,
        n_way: int,
        k_shot: int,
        q_query: int,
        download: bool,
        seed: int,
        partition: str = "all",
        val_fraction: float = 0.2,
        split_seed: int = 2026,
    ) -> None:
        if k_shot + q_query > 20:
            raise ValueError("Omniglot has 20 images per character; k_shot + q_query <= 20")
        self.dataset = datasets.Omniglot(
            root=str(root),
            background=background,
            transform=_transform(),
            download=download,
        )
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)
        self.rng = random.Random(int(seed))

        class_to_indices: dict[int, list[int]] = defaultdict(list)
        flat_items = getattr(self.dataset, "_flat_character_images", None)
        if flat_items is not None:
            for index, (_, class_index) in enumerate(flat_items):
                class_to_indices[int(class_index)].append(index)
        else:
            for index in range(len(self.dataset)):
                _, class_index = self.dataset[index]
                class_to_indices[int(class_index)].append(index)

        required = self.k_shot + self.q_query
        class_to_indices = {
            key: values for key, values in class_to_indices.items() if len(values) >= required
        }
        classes = sorted(class_to_indices)

        if partition not in ("all", "train", "validation"):
            raise ValueError("partition must be all, train, or validation")
        if partition != "all":
            if not background:
                raise ValueError("Validation must come from background characters")
            shuffled = classes.copy()
            random.Random(split_seed).shuffle(shuffled)
            n_val = max(1, round(len(shuffled) * val_fraction))
            selected = shuffled[:n_val] if partition == "validation" else shuffled[n_val:]
            classes = sorted(selected)

        if len(classes) < self.n_way:
            raise ValueError(
                f"Only {len(classes)} classes available, fewer than n_way={self.n_way}"
            )
        self.class_to_indices = class_to_indices
        self.classes = classes

    def sample(self) -> Episode:
        selected_classes = self.rng.sample(self.classes, self.n_way)
        support_x, support_y, query_x, query_y = [], [], [], []
        count = self.k_shot + self.q_query

        for episode_label, class_index in enumerate(selected_classes):
            selected = self.rng.sample(self.class_to_indices[class_index], count)
            for index in selected[: self.k_shot]:
                image, _ = self.dataset[index]
                support_x.append(image)
                support_y.append(episode_label)
            for index in selected[self.k_shot :]:
                image, _ = self.dataset[index]
                query_x.append(image)
                query_y.append(episode_label)

        return Episode(
            support_x=torch.stack(support_x),
            support_y=torch.tensor(support_y, dtype=torch.long),
            query_x=torch.stack(query_x),
            query_y=torch.tensor(query_y, dtype=torch.long),
        )

    def items(self) -> list[tuple[int, int]]:
        """Return (dataset_index, local_class_index) for supervised pretraining."""
        return [
            (index, label)
            for label, class_index in enumerate(self.classes)
            for index in self.class_to_indices[class_index]
        ]


class OmniglotPretrainDataset(Dataset):
    """Flat dataset over the training characters, used only by Fixed-2."""

    def __init__(self, sampler: OmniglotEpisodeSampler) -> None:
        self.dataset = sampler.dataset
        self.items = sampler.items()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        dataset_index, label = self.items[index]
        image, _ = self.dataset[dataset_index]
        return image, label
