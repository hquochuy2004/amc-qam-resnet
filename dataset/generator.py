"""Synthetic I/Q burst generator.

    r[n] = exp(j(2*pi*df*n + phi)) * s[n] + w[n],   w ~ CN(0, sigma^2)
    SNR  = 10 log10( mean|s|^2 / sigma^2 )
    r_norm[n] = r[n] / E_rms,   E_rms = sqrt( mean_n |r[n]|^2 )

Output per split: X float32 [N, 2, 128] (row 0 = I, row 1 = Q), y int64 class
index, snr int16. Each split draws from its own RNG seed (config.data.split_seeds),
so no burst, symbol sequence or channel draw is shared between train/val/test.

Usage:
    python -m dataset.generator            # full dataset -> data/{train,val,test}.npz
    python -m dataset.generator --quick    # 20 bursts per (class, SNR), for run_all --quick
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from utils.config import ROOT, load_config

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- pulse shapes
def rrc_taps(sps: int, beta: float, span: int) -> np.ndarray:
    """Root-raised-cosine taps, length span*sps + 1, unit energy."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps**2))


def gaussian_taps(sps: int, bt: float, span: int = 4) -> np.ndarray:
    """Gaussian frequency-pulse filter for GFSK, unit DC gain."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    taps = np.exp(-((np.pi * t / alpha) ** 2))
    return taps / taps.sum()


# ---------------------------------------------------------------- constellations
def _qam(m: int) -> np.ndarray:
    k = int(np.sqrt(m))
    levels = np.arange(-(k - 1), k, 2)
    pts = (levels[:, None] + 1j * levels[None, :]).ravel()
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def _psk(m: int) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * np.arange(m) / m)


def _pam(m: int) -> np.ndarray:
    pts = np.arange(-(m - 1), m, 2).astype(complex)
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


CONSTELLATIONS = {
    "BPSK": _psk(2),
    "QPSK": _psk(4) * np.exp(1j * np.pi / 4),
    "8PSK": _psk(8),
    "16QAM": _qam(16),
    "64QAM": _qam(64),
    "PAM4": _pam(4),
}
FSK_CLASSES = ("CPFSK", "GFSK")


# ---------------------------------------------------------------- modulators
def _linear_mod(rng, name, n, n_sym, sps, taps):
    pts = CONSTELLATIONS[name]
    sym = pts[rng.integers(0, len(pts), size=(n, n_sym))]
    up = np.zeros((n, n_sym * sps), dtype=complex)
    up[:, ::sps] = sym
    return fftconvolve(up, taps[None, :], mode="full", axes=1)


def _fsk_mod(rng, name, n, n_sym, sps, h, gauss):
    """Binary continuous-phase FSK; GFSK smooths the frequency pulse first."""
    bits = rng.integers(0, 2, size=(n, n_sym)) * 2 - 1
    freq = np.repeat(bits, sps, axis=1).astype(float)
    if name == "GFSK":
        freq = fftconvolve(freq, gauss[None, :], mode="same", axes=1)
    phase = np.cumsum(np.pi * h * freq / sps, axis=1)
    return np.exp(1j * phase)


def modulate(rng, name, n, cfg, ideal=False):
    """Return n clean complex baseband bursts of length L, shape [n, L].

    ideal=True removes the random timing offset (used for sanity plots).
    """
    d = cfg["data"]
    L, sps, span = d["burst_len"], d["sps"], d["rrc_span"]
    n_sym = L // sps + 2 * span
    if name in CONSTELLATIONS:
        tx = _linear_mod(rng, name, n, n_sym, sps, rrc_taps(sps, d["rrc_rolloff"], span))
    elif name in FSK_CLASSES:
        tx = _fsk_mod(rng, name, n, n_sym, sps, d["fsk_mod_index"], gaussian_taps(sps, d["gfsk_bt"]))
    else:
        raise ValueError(f"unknown modulation {name}")
    # skip the filter transient, then a random timing offset within one symbol
    off = np.zeros(n, dtype=int) if ideal else rng.integers(0, sps, size=n)
    idx = (span * sps + off)[:, None] + np.arange(L)[None, :]
    return np.take_along_axis(tx, idx, axis=1)


# ---------------------------------------------------------------- channel
def channel(rng, s, snr_db, cfg, ideal=False, return_parts=False):
    """Phase offset, CFO and AWGN at the given SNR, per burst.

    ideal=True sets phase and CFO to zero (AWGN only). return_parts=True also
    returns the rotated clean signal and the noise, for measuring the real SNR.
    """
    n, L = s.shape
    d = cfg["data"]
    if ideal:
        phi, df = np.zeros((n, 1)), np.zeros((n, 1))
    else:
        phi = rng.uniform(0, 2 * np.pi, size=(n, 1))
        df = rng.uniform(-d["cfo_max"], d["cfo_max"], size=(n, 1))
    s = s * np.exp(1j * (2 * np.pi * df * np.arange(L)[None, :] + phi))
    p_sig = np.mean(np.abs(s) ** 2, axis=1, keepdims=True)
    sigma2 = p_sig / (10 ** (snr_db / 10))
    w = np.sqrt(sigma2 / 2) * (rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L)))
    r = s + w
    return (r, s, w) if return_parts else r


def rms_normalize(r: np.ndarray) -> np.ndarray:
    return r / np.sqrt(np.mean(np.abs(r) ** 2, axis=1, keepdims=True))


def to_tensor(r: np.ndarray) -> np.ndarray:
    """Complex [N, L] -> float32 [N, 2, L] with row 0 = I, row 1 = Q."""
    return np.stack([r.real, r.imag], axis=1).astype(np.float32)


# ---------------------------------------------------------------- dataset
def generate_split(cfg, split: str, n_per: int | None = None):
    """Generate one split from its own seed. Returns X [N, 2, L], y [N], snr [N]."""
    d = cfg["data"]
    n_per = n_per or d["bursts_per_class_snr"][split]
    rng = np.random.default_rng(d["split_seeds"][split])
    X, y, snr = [], [], []
    for ci, name in enumerate(d["classes"]):
        for s_db in d["snr_db"]:
            r = rms_normalize(channel(rng, modulate(rng, name, n_per, cfg), s_db, cfg))
            X.append(to_tensor(r))
            y.append(np.full(n_per, ci, dtype=np.int64))
            snr.append(np.full(n_per, s_db, dtype=np.int16))
    return np.concatenate(X), np.concatenate(y), np.concatenate(snr)


def save_all(cfg, out_dir: Path, n_per: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        t0 = time.time()
        X, y, snr = generate_split(cfg, split, n_per)
        np.savez(out_dir / f"{split}.npz", X=X, y=y, snr=snr, classes=np.array(cfg["data"]["classes"]))
        print(f"{split:5s}: X {X.shape} -> {out_dir / (split + '.npz')}  ({time.time() - t0:.1f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small dataset for smoke tests and the demo")
    ap.add_argument("--out", default=None, help="output folder (default: config data.out_dir)")
    args = ap.parse_args()
    cfg = load_config()
    n_per = cfg["quick"]["bursts_per_class_snr"] if args.quick else None
    out = Path(args.out) if args.out else ROOT / cfg["data"]["out_dir"]
    save_all(cfg, out, n_per)


if __name__ == "__main__":
    main()
