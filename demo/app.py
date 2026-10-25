"""Streamlit QAM Test Bench: `streamlit run demo/app.py` from the repo root.

Generates fresh bursts with the same generator and channel as the dataset (new
RNG seeds, so nothing here was seen in training) and runs the three trained
models side by side on CPU.

Tabs:
    Single burst  - constellation, I/Q trace and the three models' class probabilities
    Batch test    - accuracy and the 16QAM/64QAM 2x2 block on N new bursts at one SNR
    Saved results - figures and tables produced by evaluate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from dataset.generator import channel, modulate, rms_normalize, to_tensor  # noqa: E402
from models import MODEL_NAMES, build_model, count_params  # noqa: E402
from utils.config import load_config  # noqa: E402

LABELS = {"cnn_base": "CNN-Base", "cnn_plain_deep": "CNN-Deep-Plain", "resnet1d": "ResNet1D"}
COLORS = {"cnn_base": "#7f7f7f", "cnn_plain_deep": "#1f77b4", "resnet1d": "#d62728"}
DEMO_SEED_BASE = 10_000  # far from the dataset split seeds (1000/2000/3000)

st.set_page_config(page_title="QAM Test Bench", layout="wide")
cfg = load_config()
CLASSES = cfg["data"]["classes"]
SNRS = cfg["data"]["snr_db"]


@st.cache_resource
def load_models(seed: int):
    out = {}
    for m in MODEL_NAMES:
        ck = ROOT / cfg["train"]["ckpt_dir"] / f"{m}_s{seed}.pt"
        if not ck.exists():
            continue
        state = torch.load(ck, map_location="cpu")
        model = build_model(m, cfg)
        model.load_state_dict(state["state_dict"])
        out[m] = model.eval()
    return out


def make_bursts(name, n, snr_db, rng, impairments=True):
    s = modulate(rng, name, n, cfg, ideal=not impairments)
    r = rms_normalize(channel(rng, s, snr_db, cfg, ideal=not impairments))
    return r, to_tensor(r)


@torch.no_grad()
def predict_proba(model, X):
    return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("QAM Test Bench")
seed = st.sidebar.selectbox("Trained seed", cfg["train"]["model_seeds"], index=0)
models = load_models(seed)
if not models:
    st.error(f"No checkpoints found in {ROOT / cfg['train']['ckpt_dir']}. Run train.py first.")
    st.stop()
st.sidebar.caption(" · ".join(f"{LABELS[m]} {count_params(v) / 1e3:.0f}k" for m, v in models.items()))
impair = st.sidebar.checkbox("Random phase + CFO", value=True)

tab1, tab2, tab3 = st.tabs(["Single burst", "Batch test", "Saved results"])

# ---------------------------------------------------------------- single burst
with tab1:
    c1, c2, c3 = st.columns([2, 3, 1])
    mod = c1.selectbox("Modulation", CLASSES, index=CLASSES.index("64QAM"))
    snr = c2.select_slider("SNR (dB)", SNRS, value=14)
    if "draw" not in st.session_state:
        st.session_state.draw = 0
    if c3.button("New burst", use_container_width=True):
        st.session_state.draw += 1
    rng = np.random.default_rng(DEMO_SEED_BASE + st.session_state.draw)
    r, X = make_bursts(mod, 1, snr, rng, impair)

    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(r[0].real, r[0].imag, s=10, c=np.arange(r.shape[1]), cmap="viridis")
    lim = 1.1 * np.abs(r[0]).max()
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="I", ylabel="Q", aspect="equal",
           title=f"{mod} at {snr:+d} dB (128 samples)")
    ax.grid(alpha=0.3)
    left.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.plot(r[0].real, lw=1, label="I")
    ax.plot(r[0].imag, lw=1, label="Q")
    ax.set(xlabel="sample", title="I/Q trace")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    right.pyplot(fig, clear_figure=True)

    probs = {LABELS[m]: predict_proba(v, X)[0] for m, v in models.items()}
    rows = []
    for name, p in probs.items():
        k = int(p.argmax())
        rows.append({"Model": name, "Prediction": CLASSES[k], "Confidence": f"{p[k]:.1%}",
                     f"P({mod})": f"{p[CLASSES.index(mod)]:.1%}", "Correct": "✅" if CLASSES[k] == mod else "❌"})
    right.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 2.6))
    w = 0.8 / len(probs)
    for i, (m, p) in enumerate(zip(models, probs.values())):
        ax.bar(np.arange(len(CLASSES)) + (i - 1) * w, p, w, label=LABELS[m], color=COLORS[m])
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.get_xticklabels()[CLASSES.index(mod)].set_fontweight("bold")
    ax.set(ylim=(0, 1), ylabel="probability", title="Class probabilities (true class in bold)")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------- batch test
with tab2:
    st.markdown("Fresh bursts at one SNR, all three models on the same bursts.")
    c1, c2, c3 = st.columns([3, 2, 1])
    picked = c1.multiselect("Classes", CLASSES, default=["16QAM", "64QAM"])
    bsnr = c2.select_slider("SNR (dB) ", SNRS, value=14)
    n = c3.number_input("Bursts / class", 50, 2000, 300, step=50)
    if st.button("Run batch", type="primary") and picked:
        rng = np.random.default_rng(DEMO_SEED_BASE + 777 + bsnr)
        Xs, ys = [], []
        for name in picked:
            Xs.append(make_bursts(name, n, bsnr, rng, impair)[1])
            ys.append(np.full(n, CLASSES.index(name)))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        preds = {m: predict_proba(v, X).argmax(1) for m, v in models.items()}

        cols = st.columns(len(preds))
        for col, (m, p) in zip(cols, preds.items()):
            col.metric(LABELS[m], f"{np.mean(p == y):.1%}")
            per = {name: np.mean(p[y == CLASSES.index(name)] == CLASSES.index(name)) for name in picked}
            col.dataframe(pd.DataFrame({"class": per.keys(), "accuracy": [f"{v:.1%}" for v in per.values()]}),
                          hide_index=True, use_container_width=True)

        if {"16QAM", "64QAM"} <= set(picked):
            st.subheader("16QAM / 64QAM block (row = true class)")
            i16, i64 = CLASSES.index("16QAM"), CLASSES.index("64QAM")
            cols = st.columns(len(preds))
            for col, (m, p) in zip(cols, preds.items()):
                block = [[np.mean(p[y == t] == q) for q in (i16, i64)] for t in (i16, i64)]
                col.markdown(f"**{LABELS[m]}**")
                col.dataframe(pd.DataFrame(block, index=["true 16QAM", "true 64QAM"],
                                           columns=["→ 16QAM", "→ 64QAM"]).style.format("{:.1%}"),
                              use_container_width=True)

# ---------------------------------------------------------------- saved results
with tab3:
    res = ROOT / cfg["eval"]["results_dir"]
    summ = res / "summary.csv"
    if summ.exists():
        df = pd.read_csv(summ)
        agg = df.groupby("model")[["params", "acc_all", "macro_f1", "acc_high_snr", "latency_ms"]].agg(["mean", "std"])
        st.subheader("Test set, mean ± std over 3 seeds")
        st.dataframe(agg.round(4), use_container_width=True)
    figs = ["acc_vs_snr.png", "qam_rate_vs_snr.png", "cm_cnn_base.png", "cm_resnet1d.png", "training_curves.png"]
    for f in figs:
        p = res / "figures" / f
        if p.exists():
            st.image(str(p), caption=f, use_container_width=True)
