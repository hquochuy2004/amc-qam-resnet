"""Model checks: output shape and matched parameter count (+-10% of target)."""
import pytest
import torch

from models import MODEL_NAMES, build_model, count_params
from utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(cfg, name):
    out = build_model(name, cfg)(torch.randn(4, 2, cfg["data"]["burst_len"]))
    assert out.shape == (4, len(cfg["data"]["classes"]))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_params_matched(cfg, name):
    target = cfg["models"]["target_params"]
    assert abs(count_params(build_model(name, cfg)) - target) <= 0.1 * target


def test_plain_and_resnet_differ_only_by_skips(cfg):
    plain, res = build_model("cnn_plain_deep", cfg), build_model("resnet1d", cfg)
    assert len(plain.stages) == len(res.stages)
    assert all(not u.residual for u in plain.stages) and all(u.residual for u in res.stages)
