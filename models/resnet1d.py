"""ResNet1D: stem + 3 stages x 2 residual units, y = ReLU(F(x) + x), 1x1 projection when shape changes."""
from models.blocks import StagedNet


class ResNet1D(StagedNet):
    def __init__(self, n_classes, stem_channels, stage_channels, units_per_stage, dropout):
        super().__init__(n_classes, stem_channels, stage_channels, units_per_stage, dropout, residual=True)
