from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
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


class OmniglotEpisodeSampler:
    """从 Omniglot 字符类别中随机构造 N-way K-shot 任务。"""

    def __init__(
        self,
        root: str | Path,
        background: bool,
        n_way: int,
        k_shot: int,
        q_query: int,
        download: bool,
        seed: int,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda image: 1.0 - image),
            ]
        )
        self.dataset = datasets.Omniglot(
            root=str(root),
            background=background,
            transform=transform,
            download=download,
        )
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = random.Random(seed)

        class_to_indices: dict[int, list[int]] = defaultdict(list)
        flat_items = getattr(self.dataset, "_flat_character_images", None)
        if flat_items is not None:
            for index, (_, class_index) in enumerate(flat_items):
                class_to_indices[int(class_index)].append(index)
        else:
            for index in range(len(self.dataset)):
                _, class_index = self.dataset[index]
                class_to_indices[int(class_index)].append(index)

        required = k_shot + q_query
        self.class_to_indices = {
            key: values for key, values in class_to_indices.items() if len(values) >= required
        }
        self.classes = sorted(self.class_to_indices)
        if len(self.classes) < n_way:
            raise ValueError(
                f"可用类别数 {len(self.classes)} 小于 n_way={n_way}；"
                "请减小 n_way 或检查数据是否下载完整。"
            )

    def sample(self) -> Episode:
        selected_classes = self.rng.sample(self.classes, self.n_way)
        support_x, support_y, query_x, query_y = [], [], [], []
        count = self.k_shot + self.q_query

        for episode_label, class_index in enumerate(selected_classes):
            selected = self.rng.sample(self.class_to_indices[class_index], count)
            support_indices = selected[: self.k_shot]
            query_indices = selected[self.k_shot :]
            for index in support_indices:
                image, _ = self.dataset[index]
                support_x.append(image)
                support_y.append(episode_label)
            for index in query_indices:
                image, _ = self.dataset[index]
                query_x.append(image)
                query_y.append(episode_label)

        return Episode(
            support_x=torch.stack(support_x),
            support_y=torch.tensor(support_y, dtype=torch.long),
            query_x=torch.stack(query_x),
            query_y=torch.tensor(query_y, dtype=torch.long),
        )

