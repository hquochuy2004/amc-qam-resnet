"""Load the saved .npz splits as PyTorch tensors."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_split(data_dir: str | Path, split: str):
    """Return (TensorDataset(X, y), snr numpy array, class names)."""
    d = np.load(Path(data_dir) / f"{split}.npz")
    ds = TensorDataset(torch.from_numpy(d["X"]), torch.from_numpy(d["y"]))
    return ds, d["snr"], [str(c) for c in d["classes"]]
