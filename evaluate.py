"""Evaluate every checkpoint on the test split and write all figures and tables.

    python evaluate.py                # uses checkpoints/*.pt and data/test.npz
    python evaluate.py --data <dir>   # other data folder

Outputs in results/:
  acc_vs_snr.png, f1_vs_snr.png             E1, E2 (mean +- std over seeds)
  cm_<model>.png                            E3 at the configured SNRs (seed 0)
  qam_subblock.csv, qam_rate_vs_snr.png     E4, E5
  per_class_f1.csv                          E6 (SNR >= qam_curve_snr_min)
  training_curves.png                       E7 (from results/logs)
  summary.csv                               params, accuracy, Macro-F1, latency (E8)
  mcnemar.csv                               E9, ResNet vs the other two on QAM at high SNR
  predictions_<model>_s<seed>.npz           raw predictions for further analysis
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.torch_dataset import load_split
from models import MODEL_NAMES, build_model, count_params
from utils import metrics as M
from utils import plots as P
from utils.config import ROOT, load_config


@torch.no_grad()
def predict(model, X, device, batch=1024):
    model.eval()
    out = [model(X[i:i + batch].to(device)).argmax(1).cpu() for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


@torch.no_grad()
def latency_ms(model, X, n):
    """Single-burst CPU latency, averaged over n bursts after 20 warm-up calls."""
    model = model.cpu().eval()
    for i in range(20):
        model(X[i:i + 1])
    t0 = time.perf_counter()
    for i in range(n):
        model(X[i % len(X):i % len(X) + 1])
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    cfg = load_config()
    ev = cfg["eval"]
    data_dir = Path(args.data) if args.data else ROOT / cfg["data"]["out_dir"]
    out = ROOT / ev["results_dir"]
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds, snr, classes = load_split(data_dir, "test")
    X, y = test_ds.tensors[0], test_ds.tensors[1].numpy()
    grid = cfg["data"]["snr_db"]
    i16, i64 = classes.index(ev["qam_pair"][0]), classes.index(ev["qam_pair"][1])
    hi = snr >= ev["qam_high_snr_min"]
    qam_grid = [s for s in grid if s >= ev["qam_curve_snr_min"]]

    preds, summary, sub_rows, f1_rows = {}, [], [], []
    acc_s, f1_s, rate_s = {}, {}, {}
    for m in MODEL_NAMES:
        ckpts = sorted((ROOT / cfg["train"]["ckpt_dir"]).glob(f"{m}_s*.pt"))
        if not ckpts:
            print(f"skip {m}: no checkpoints")
            continue
        accs, f1s, rates = [], [], []
        for ck in ckpts:
            state = torch.load(ck, map_location="cpu")
            model = build_model(m, cfg)
            model.load_state_dict(state["state_dict"])
            seed = state["seed"]
            p = predict(model.to(device), X, device)
            preds[(m, seed)] = p
            np.savez(out / f"predictions_{m}_s{seed}.npz", pred=p, y=y, snr=snr)
            accs.append(M.acc_by_snr(y, p, snr, grid))
            f1s.append(M.f1_by_snr(y, p, snr, grid))
            rates.append(M.qam_confusion_rate(y, p, snr, qam_grid, i16, i64))
            b = M.qam_subblock(y[hi], p[hi], i16, i64)
            sub_rows.append(dict(model=m, seed=seed, p16_16=b[0, 0], p16_64=b[0, 1], p64_16=b[1, 0], p64_64=b[1, 1]))
            f1c = M.f1_score(y[snr >= ev["qam_curve_snr_min"]], p[snr >= ev["qam_curve_snr_min"]],
                             average=None, labels=range(len(classes)))
            f1_rows.append(dict(model=m, seed=seed, **{c: v for c, v in zip(classes, f1c)}))
            summary.append(dict(model=m, seed=seed, params=count_params(model), epoch=state["epoch"],
                                acc_all=float(np.mean(p == y)), macro_f1=M.macro_f1(y, p),
                                acc_high_snr=float(np.mean(p[hi] == y[hi])),
                                latency_ms=latency_ms(model, X, ev["latency_bursts"])))
            if seed == min(s for (mm, s) in preds if mm == m):
                cms = [M.cm_row_norm(y[snr == s], p[snr == s], len(classes)) for s in ev["cm_snrs"]]
                P.confusion_grid(cms, classes, [f"{s:+d} dB" for s in ev["cm_snrs"]],
                                 fig_dir / f"cm_{m}.png", f"{P.MODEL_STYLE[m]['label']}: row-normalized confusion")
            print(f"{ck.name}: acc {summary[-1]['acc_all']:.4f}  macro-F1 {summary[-1]['macro_f1']:.4f}  "
                  f"acc(SNR>={ev['qam_high_snr_min']}) {summary[-1]['acc_high_snr']:.4f}")
        acc_s[m] = (np.mean(accs, 0), np.std(accs, 0))
        f1_s[m] = (np.mean(f1s, 0), np.std(f1s, 0))
        rate_s[m] = (np.mean(rates, 0), np.std(rates, 0))

    if not preds:
        raise SystemExit("no checkpoints found; run train.py first")
    P.curves_vs_snr(grid, acc_s, "Accuracy", "Accuracy vs SNR (mean ± std over seeds)", fig_dir / "acc_vs_snr.png")
    P.curves_vs_snr(grid, f1_s, "Macro-F1", "Macro-F1 vs SNR", fig_dir / "f1_vs_snr.png")
    P.curves_vs_snr(qam_grid, rate_s, "Mean 16QAM↔64QAM confusion",
                    "16QAM/64QAM confusion at positive SNR", fig_dir / "qam_rate_vs_snr.png", ylim=(0, 0.6))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(sub_rows).to_csv(out / "qam_subblock.csv", index=False)
    pd.DataFrame(f1_rows).to_csv(out / "per_class_f1.csv", index=False)
    pd.DataFrame([dict(snr_db=s, **{m: acc_s[m][0][k] for m in acc_s}) for k, s in enumerate(grid)]).to_csv(
        out / "acc_vs_snr.csv", index=False)

    # E9: McNemar on QAM bursts at high SNR, ResNet vs the others, matched seeds
    q = hi & ((y == i16) | (y == i64))
    rows = []
    for other in ("cnn_base", "cnn_plain_deep"):
        for (m, s), p in preds.items():
            if m == "resnet1d" and (other, s) in preds:
                n01, n10, pv = M.mcnemar(y[q], p[q], preds[(other, s)][q])
                rows.append(dict(seed=s, vs=other, resnet_right_other_wrong=n01, resnet_wrong_other_right=n10, p_value=pv))
    with open(out / "mcnemar.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # E7: training curves from the logs written by train.py
    logs = {m: [pd.read_csv(f) for f in sorted((out / "logs").glob(f"{m}_s*.csv"))] for m in MODEL_NAMES}
    logs = {m: v for m, v in logs.items() if v}
    if logs:
        P.training_curves(logs, fig_dir / "training_curves.png")
    print(pd.DataFrame(summary).groupby("model")[["acc_all", "macro_f1", "acc_high_snr", "latency_ms"]].agg(["mean", "std"]).round(4))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
