"""A small convolutional network for 28x28 Omniglot images."""

from __future__ import annotations

import torch
from torch import nn


class OmniglotConvNet(nn.Module):
    """Four conv/pool blocks followed by a linear classifier.

    No BatchNorm is used: the network is evaluated through
    ``torch.func.functional_call`` during inner-loop adaptation, and keeping
    the module free of running buffers makes that swap unambiguous.
    """

    def __init__(self, num_classes: int, hidden_channels: int = 32) -> None:
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
        self.feature_dim = hidden_channels
        self.classifier = nn.Linear(self.feature_dim, int(num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = torch.flatten(features, start_dim=1)
        return self.classifier(features)

    def reset_classifier(self, num_classes: int) -> None:
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        self.classifier = nn.Linear(self.feature_dim, int(num_classes)).to(
            device=device, dtype=dtype
        )
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)
