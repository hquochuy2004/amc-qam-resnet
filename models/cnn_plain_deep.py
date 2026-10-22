"""CNN-Deep-Plain: exactly the ResNet1D layout with the skip connections removed (ablation)."""
from models.blocks import StagedNet


class CNNPlainDeep(StagedNet):
    def __init__(self, n_classes, stem_channels, stage_channels, units_per_stage, dropout):
        super().__init__(n_classes, stem_channels, stage_channels, units_per_stage, dropout, residual=False)
