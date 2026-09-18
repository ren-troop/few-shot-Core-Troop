from __future__ import annotations

import torch
from torch import nn


class OmniglotConvNet(nn.Module):
    """不使用 BatchNorm，便于进行函数式参数更新。"""

    def __init__(self, n_way: int, hidden_channels: int = 32) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for _ in range(4):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = hidden_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_channels, n_way)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = features.flatten(start_dim=1)
        return self.classifier(features)

