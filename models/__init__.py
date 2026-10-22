"""Model factory: build_model(name, cfg)."""
from models.cnn_base import CNNBase
from models.cnn_plain_deep import CNNPlainDeep
from models.resnet1d import ResNet1D

MODEL_NAMES = ("cnn_base", "cnn_plain_deep", "resnet1d")


def build_model(name: str, cfg: dict):
    m, n_classes = cfg["models"], len(cfg["data"]["classes"])
    if name == "cnn_base":
        return CNNBase(n_classes, dropout=m["dropout"], **m["cnn_base"])
    if name == "cnn_plain_deep":
        return CNNPlainDeep(n_classes, dropout=m["dropout"], **m["cnn_plain_deep"])
    if name == "resnet1d":
        return ResNet1D(n_classes, dropout=m["dropout"], **m["resnet1d"])
    raise ValueError(f"unknown model {name}, expected one of {MODEL_NAMES}")


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
