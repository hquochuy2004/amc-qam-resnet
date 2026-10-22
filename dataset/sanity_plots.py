"""Gate-1 sanity plots and measured-SNR check.

Run: python -m dataset.sanity_plots      -> results/figures/*.png, results/snr_check.csv
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erfc

from dataset.generator import CONSTELLATIONS, channel, modulate, rms_normalize, rrc_taps
from utils.config import ROOT, load_config

BLUE, DEEP, INK, GRID = "#0284C7", "#2563EB", "#0F172A", "#E2E8F0"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "font.size": 10, "figure.dpi": 150})


def matched_symbols(cfg, r):
    """Matched RRC filter + symbol-rate sampling (valid only for ideal timing)."""
    d = cfg["data"]
    sps, span = d["sps"], d["rrc_span"]
    y = fftconvolve(r, rrc_taps(sps, d["rrc_rolloff"], span)[None, :], mode="same", axes=1)
    edge = span // 2
    return y[:, ::sps][:, edge:-edge]


def c42(x):
    """Normalized 4th-order cumulant C42 per row."""
    x = x - x.mean(axis=1, keepdims=True)
    m2 = np.mean(np.abs(x) ** 2, axis=1)
    m4 = np.mean(np.abs(x) ** 4, axis=1)
    m20 = np.mean(x**2, axis=1)
    return (m4 - np.abs(m20) ** 2 - 2 * m2**2) / m2**2


def plot_raw(cfg, rng, out):
    classes = cfg["data"]["classes"]
    fig, axes = plt.subplots(2, len(classes), figsize=(2.1 * len(classes), 5.2))
    for row, snr in enumerate([18, 0]):
        for ci, name in enumerate(classes):
            r = rms_normalize(channel(rng, modulate(rng, name, 1, cfg), snr, cfg))[0]
            ax = axes[row, ci]
            ax.scatter(r.real, r.imag, s=4, c=BLUE)
            ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[], yticks=[], aspect="equal")
            ax.set_title(f"{name} @ {snr:+d} dB", fontsize=9, color=INK)
    fig.suptitle("Raw input bursts as the model sees them (128 samples, phase + CFO + AWGN)", color=INK)
    fig.tight_layout()
    fig.savefig(out / "sanity_raw_constellations.png")
    plt.close(fig)


def plot_qam_lattice(cfg, rng, out):
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ci, name in enumerate(["16QAM", "64QAM"]):
        for row, snr in enumerate([40, 18]):
            r = channel(rng, modulate(rng, name, 40, cfg, ideal=True), snr, cfg, ideal=True)
            sym = matched_symbols(cfg, r).ravel()
            sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2))
            ax = axes[row, ci]
            ax.scatter(sym.real, sym.imag, s=3, c=BLUE, alpha=0.6, label="matched-filter output")
            ref = CONSTELLATIONS[name]
            ax.scatter(ref.real, ref.imag, s=40, facecolors="none", edgecolors=INK, linewidths=0.8,
                       label="ideal lattice")
            ax.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), aspect="equal", xticks=[], yticks=[])
            ax.set_title(f"{name} @ {snr} dB (ideal sync, 40 bursts)", fontsize=9, color=INK)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sanity_qam_lattice.png")
    plt.close(fig)


def plot_qam_pair(cfg, rng, out, snr=18):
    """16QAM vs 64QAM side by side at one SNR, for the Background slide."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    for ax, name in zip(axes, ["16QAM", "64QAM"]):
        r = channel(rng, modulate(rng, name, 40, cfg, ideal=True), snr, cfg, ideal=True)
        sym = matched_symbols(cfg, r).ravel()
        sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2))
        ax.scatter(sym.real, sym.imag, s=3, c=BLUE, alpha=0.6)
        ref = CONSTELLATIONS[name]
        ax.scatter(ref.real, ref.imag, s=30, facecolors="none", edgecolors=INK, linewidths=0.8)
        ax.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{name} @ +{snr} dB", color=INK)
    fig.tight_layout()
    fig.savefig(out / "background_qam_pair.png", dpi=200)
    plt.close(fig)


def check_snr(cfg, rng, out, n=500):
    """Measured SNR = mean|s|^2 / mean|w|^2 over n bursts per class, vs nominal."""
    rows = []
    for snr in cfg["data"]["snr_db"]:
        ps, pw = [], []
        for name in cfg["data"]["classes"]:
            _, s, w = channel(rng, modulate(rng, name, n, cfg), snr, cfg, return_parts=True)
            ps.append(np.mean(np.abs(s) ** 2))
            pw.append(np.mean(np.abs(w) ** 2))
        meas = 10 * np.log10(np.mean(ps) / np.mean(pw))
        rows.append((snr, round(float(meas), 3), round(float(meas - snr), 3)))
    with open(out.parent / "snr_check.csv", "w", newline="") as f:
        csv.writer(f).writerows([("nominal_db", "measured_db", "error_db"), *rows])
    nom = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(nom, nom, color=GRID, lw=2, label="ideal")
    ax.plot(nom, [r[1] for r in rows], "o", color=DEEP, label="measured")
    ax.set(xlabel="Nominal SNR (dB)", ylabel="Measured SNR (dB)", title="Measured vs nominal SNR")
    ax.grid(color=GRID)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sanity_snr_measured.png")
    plt.close(fig)
    return rows


def _q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def ser_theory(name, esn0):
    """Symbol error rate of coherent detection in AWGN (Proakis)."""
    if name == "QPSK":
        q = _q(np.sqrt(esn0))
        return 2 * q - q**2
    m = int(name.replace("QAM", ""))
    p = 2 * (1 - 1 / np.sqrt(m)) * _q(np.sqrt(3 / (m - 1) * esn0))
    return 1 - (1 - p) ** 2


def check_ser(cfg, rng, out, n=2000):
    """Independent physics check: measured SER of a genie-aided coherent receiver vs theory.

    SNR in this project is per sample. With unit-energy RRC taps each symbol is spread
    over sps samples, so Es/N0 = SNR + 10 log10(sps).
    """
    sps = cfg["data"]["sps"]
    snrs = np.array(cfg["data"]["snr_db"])
    rows, fig, ax = [], *plt.subplots(figsize=(6, 4))
    for name, col in [("QPSK", BLUE), ("16QAM", DEEP), ("64QAM", INK)]:
        ref = CONSTELLATIONS[name]
        meas = []
        for snr in snrs:
            # ideal timing/phase so the receiver knows the symbol grid; draw known symbols
            d = cfg["data"]
            L, span = d["burst_len"], d["rrc_span"]
            n_sym = L // sps + 2 * span
            sym = ref[rng.integers(0, len(ref), size=(n, n_sym))]
            up = np.zeros((n, n_sym * sps), dtype=complex)
            up[:, ::sps] = sym
            tx = fftconvolve(up, rrc_taps(sps, d["rrc_rolloff"], span)[None, :], mode="full", axes=1)
            s = tx[:, span * sps: span * sps + L]
            r = channel(rng, s, snr, cfg, ideal=True)
            y = matched_symbols(cfg, r)
            # window sample k = symbol (k / sps + span / 2); matched_symbols drops span // 2 at each edge
            first = span // 2 + span // 2
            sent = sym[:, first: first + y.shape[1]]
            dec = ref[np.argmin(np.abs(y[..., None] - ref[None, None, :]), axis=-1)]
            meas.append(np.mean(dec != sent))
        meas = np.array(meas)
        esn0_db = snrs + 10 * np.log10(sps)
        fine = np.linspace(esn0_db.min(), esn0_db.max(), 200)
        ax.semilogy(fine - 10 * np.log10(sps), ser_theory(name, 10 ** (fine / 10)), color=col, lw=1.2)
        ok = meas > 0
        ax.semilogy(snrs[ok], meas[ok], "o", color=col, ms=4, label=f"{name}")
        rows += [(name, int(a), float(b), float(ser_theory(name, 10 ** (e / 10))))
                 for a, b, e in zip(snrs, meas, esn0_db)]
    ax.set(xlabel="SNR per sample (dB)   [Es/N0 = SNR + 6 dB]", ylabel="Symbol error rate",
           title="Generator matches AWGN theory (dots = measured, lines = theory)", ylim=(1e-4, 1))
    ax.grid(color=GRID, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sanity_ser_vs_theory.png")
    plt.close(fig)
    with open(out.parent / "ser_check.csv", "w", newline="") as f:
        csv.writer(f).writerows([("mod", "snr_db", "ser_measured", "ser_theory"), *rows])
    return rows


def plot_c42(cfg, rng, out, n=2000):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    stats = {}
    for name, col, theory in [("16QAM", BLUE, -0.68), ("64QAM", DEEP, -0.619)]:
        r = channel(rng, modulate(rng, name, n, cfg, ideal=True), 18, cfg, ideal=True)
        v = c42(matched_symbols(cfg, r))
        stats[name] = (float(v.mean()), float(v.std()))
        ax.hist(v, bins=60, alpha=0.55, color=col, label=f"{name} (theory {theory})")
        ax.axvline(theory, color=col, ls="--")
    n_sym = matched_symbols(cfg, r).shape[1]
    ax.set(xlabel="Empirical C42 per burst", ylabel="Bursts",
           title=f"C42 from {n_sym} symbols per burst at +18 dB overlaps heavily")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "background_c42_overlap.png")
    plt.close(fig)
    return stats


def main():
    cfg = load_config()
    out = ROOT / cfg["eval"]["results_dir"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)  # plotting only, independent of dataset split seeds
    plot_raw(cfg, rng, out)
    plot_qam_lattice(cfg, rng, out)
    plot_qam_pair(cfg, rng, out)
    rows = check_snr(cfg, rng, out)
    stats = plot_c42(cfg, rng, out)
    ser = check_ser(cfg, rng, out)
    print("SNR check (nominal, measured, error dB):")
    for r in rows:
        print(f"  {r[0]:+3d}  {r[1]:+7.3f}  {r[2]:+.3f}")
    print("max |error| = %.3f dB" % max(abs(r[2]) for r in rows))
    for k, (m, s) in stats.items():
        print(f"C42 {k}: mean {m:.3f}, std {s:.3f}")
    print("SER check (mod, SNR, measured, theory) at -4, +4, +14 dB:")
    for m, snr, a, b in ser:
        if snr in (-4, 4, 14):
            print(f"  {m:6s} {snr:+3d}  {a:.4f}  {b:.4f}")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
