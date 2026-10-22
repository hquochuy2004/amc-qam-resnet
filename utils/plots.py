"""Figures for the slides, in the course palette."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, DEEP, INK, GRID, MUTED = "#0284C7", "#2563EB", "#0F172A", "#E2E8F0", "#94A3B8"
MODEL_STYLE = {
    "cnn_base": dict(color=MUTED, label="CNN-Base"),
    "cnn_plain_deep": dict(color=BLUE, label="CNN-Deep-Plain"),
    "resnet1d": dict(color=INK, label="ResNet1D"),
}
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "font.size": 10, "figure.dpi": 150})


def curves_vs_snr(snr_grid, series, ylabel, title, path, ylim=(0, 1)):
    """series: {model: (mean[], std[])}."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, (mu, sd) in series.items():
        st = MODEL_STYLE[m]
        ax.plot(snr_grid, mu, "o-", ms=3.5, lw=1.6, color=st["color"], label=st["label"])
        ax.fill_between(snr_grid, mu - sd, mu + sd, color=st["color"], alpha=0.15, lw=0)
    ax.set(xlabel="SNR per sample (dB)", ylabel=ylabel, title=title, ylim=ylim, xticks=snr_grid)
    ax.grid(color=GRID)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_grid(cms, classes, titles, path, suptitle):
    """cms: list of row-normalized matrices, drawn side by side."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, cm, t in zip(axes, cms, titles):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if cm[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=60, fontsize=7.5)
        ax.set_yticks(range(len(classes)), classes, fontsize=7.5)
        ax.set_xlabel("Predicted")
        ax.set_title(t, fontsize=10)
    axes[0].set_ylabel("True")
    fig.suptitle(suptitle, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def training_curves(logs, path):
    """logs: {model: list of DataFrames (one per seed)}; val loss and stem grad norm."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for m, dfs in logs.items():
        st = MODEL_STYLE[m]
        for k, df in enumerate(dfs):
            lab = st["label"] if k == 0 else None
            axes[0].plot(df["epoch"], df["val_loss"], color=st["color"], lw=1.2, label=lab)
            axes[1].plot(df["epoch"], df["stem_grad_norm"], color=st["color"], lw=1.2, label=lab)
    axes[0].set(xlabel="Epoch", ylabel="Validation loss", title="Convergence")
    axes[1].set(xlabel="Epoch", ylabel="Gradient norm at first conv", title="Gradient reaching the input layer",
                yscale="log")
    for ax in axes:
        ax.grid(color=GRID)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
