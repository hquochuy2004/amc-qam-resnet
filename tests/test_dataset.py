"""Gate-1 checks: tensor shape, RMS normalization, no leakage, determinism.

Run: python -m pytest -q
"""
import hashlib

import numpy as np
import pytest

from dataset.generator import SPLITS, generate_split
from utils.config import load_config

N_PER = 10  # bursts per (class, SNR) for the tests


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def splits(cfg):
    return {s: generate_split(cfg, s, N_PER) for s in SPLITS}


def test_shape_and_dtype(cfg, splits):
    n_cls, n_snr = len(cfg["data"]["classes"]), len(cfg["data"]["snr_db"])
    for X, y, snr in splits.values():
        assert X.shape == (n_cls * n_snr * N_PER, 2, cfg["data"]["burst_len"])
        assert X.dtype == np.float32 and y.dtype == np.int64
        assert np.isfinite(X).all()


def test_rms_is_one(splits):
    for X, _, _ in splits.values():
        rms = np.sqrt(np.mean(X[:, 0] ** 2 + X[:, 1] ** 2, axis=1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-5)


def test_balanced_labels_and_snr_grid(cfg, splits):
    for _, y, snr in splits.values():
        assert np.all(np.bincount(y) == len(cfg["data"]["snr_db"]) * N_PER)
        assert sorted(np.unique(snr).tolist()) == cfg["data"]["snr_db"]


def test_no_duplicate_bursts_across_splits(splits):
    hashes = {s: {hashlib.sha1(x.tobytes()).hexdigest() for x in X} for s, (X, _, _) in splits.items()}
    for s, h in hashes.items():
        assert len(h) == len(splits[s][0]), f"duplicate bursts inside {s}"
    assert not hashes["train"] & hashes["val"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["val"] & hashes["test"]


def test_split_seeds_are_distinct(cfg):
    seeds = list(cfg["data"]["split_seeds"].values())
    assert len(set(seeds)) == len(seeds)


def test_deterministic(cfg, splits):
    X2, y2, _ = generate_split(cfg, "test", N_PER)
    np.testing.assert_array_equal(X2, splits["test"][0])
    np.testing.assert_array_equal(y2, splits["test"][1])
