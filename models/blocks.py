"""Shared 1D building blocks."""
import torch.nn as nn


def conv_bn(c_in: int, c_out: int, k: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm1d(c_out),
    )


class Unit(nn.Module):
    """Two 3-tap convs. residual=True gives y = ReLU(F(x) + x); False gives the plain y = ReLU(F(x))."""

    def __init__(self, c_in: int, c_out: int, stride: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(conv_bn(c_in, c_out, 3, stride), nn.ReLU(inplace=True), conv_bn(c_out, c_out, 3))
        self.act = nn.ReLU(inplace=True)
        self.shortcut = None
        if residual and (stride != 1 or c_in != c_out):
            # 1x1 projection when the shape changes, identity otherwise
            self.shortcut = conv_bn(c_in, c_out, 1, stride)

    def forward(self, x):
        y = self.f(x)
        if self.residual:
            y = y + (x if self.shortcut is None else self.shortcut(x))
        return self.act(y)


class StagedNet(nn.Module):
    """Stem + stages of Units + global average pooling + linear head.

    Shared by ResNet1D (residual=True) and CNNPlainDeep (residual=False) so the
    two differ only in the skip connections.
    """

    def __init__(self, n_classes, stem_channels, stage_channels, units_per_stage, dropout, residual):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(2, stem_channels, 7), nn.ReLU(inplace=True))
        layers, c = [], stem_channels
        for si, c_out in enumerate(stage_channels):
            for ui in range(units_per_stage):
                stride = 2 if (si > 0 and ui == 0) else 1  # downsample 128 -> 64 -> 32
                layers.append(Unit(c, c_out, stride, residual))
                c = c_out
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))

    def embed(self, x):
        """Pooled features before the classifier (for t-SNE / analysis)."""
        return self.head[1](self.head[0](self.stages(self.stem(x))))
