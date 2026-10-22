"""CNN-Base: shallow feedforward 1D CNN, 4 Conv-BN-ReLU-MaxPool blocks, global average pooling, linear head."""
import torch.nn as nn

from models.blocks import conv_bn


class CNNBase(nn.Module):
    def __init__(self, n_classes, channels, kernels, dropout):
        super().__init__()
        layers, c = [], 2
        for c_out, k in zip(channels, kernels):
            layers += [conv_bn(c, c_out, k), nn.ReLU(inplace=True), nn.MaxPool1d(2)]
            c = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def embed(self, x):
        return self.head[1](self.head[0](self.features(x)))
