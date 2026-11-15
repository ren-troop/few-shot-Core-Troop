"""Backbone networks used by the four algorithms.

The default is the standard four-layer convolutional network ("Conv4") used in
the original MAML paper.  Group normalization is the default because it has no
running buffers: the same functional forward therefore behaves identically
during inner-loop adaptation and during query evaluation.  BatchNorm is kept as
an option for users who want to reproduce a specific paper exactly.
"""

from __future__ import annotations

import torch
from torch import nn


def _norm_layer(kind: str, channels: int, groups: int = 8) -> nn.Module:
    kind = kind.lower()
    if kind == "group":
        g = groups if channels % groups == 0 else 1
        return nn.GroupNorm(g, channels)
    if kind == "batch":
        # track_running_stats=False means "use the current batch statistics".
        # This avoids stale running statistics in episodic training.
        return nn.BatchNorm2d(channels, track_running_stats=False)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm = _norm_layer(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.norm(self.conv(x))))


class Conv4(nn.Module):
    """Four conv/pool blocks followed by a linear classifier.

    For a 32x32 input the feature map after four pools is 2x2, so the flattened
    feature size is ``hidden * 4``.  For an 84x84 input it is 5x5, giving
    ``hidden * 25``.  The model is deliberately resolution-aware rather than
    using global pooling, matching the usual MAML Conv4 implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden: int = 64,
        num_classes: int = 5,
        image_size: int = 32,
        norm: str = "group",
    ):
        super().__init__()
        if image_size < 16:
            raise ValueError("image_size must be at least 16 for four 2x pools")
        self.image_size = int(image_size)
        self.hidden = int(hidden)
        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden, norm),
            ConvBlock(hidden, hidden, norm),
            ConvBlock(hidden, hidden, norm),
            ConvBlock(hidden, hidden, norm),
        )
        spatial = self.image_size // 16
        self.feature_dim = hidden * spatial * spatial
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the classifier head, e.g. for a new N-way episode."""
        device = self.classifier.weight.device
        dtype = self.classifier.weight.dtype
        self.classifier = nn.Linear(self.feature_dim, int(num_classes)).to(
            device=device, dtype=dtype
        )
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)


def build_model(cfg, num_classes: int) -> Conv4:
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    return Conv4(
        in_channels=int(model_cfg.get("in_channels", 3)),
        hidden=int(model_cfg.get("hidden", 64)),
        num_classes=int(num_classes),
        image_size=int(dataset_cfg.get("image_size", 32)),
        norm=str(model_cfg.get("norm", "group")),
    )
