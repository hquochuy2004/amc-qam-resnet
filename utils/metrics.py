"""Metrics: accuracy vs SNR, Macro-F1, row-normalized confusion matrices, QAM sub-block, McNemar."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score


def acc_by_snr(y, pred, snr, snr_grid):
    return np.array([np.mean(pred[snr == s] == y[snr == s]) for s in snr_grid])


def f1_by_snr(y, pred, snr, snr_grid):
    return np.array([f1_score(y[snr == s], pred[snr == s], average="macro") for s in snr_grid])


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro")


def cm_row_norm(y, pred, n_classes):
    cm = confusion_matrix(y, pred, labels=range(n_classes)).astype(float)
    return cm / cm.sum(axis=1, keepdims=True).clip(min=1)


def qam_subblock(y, pred, i16, i64):
    """2x2 row-normalized block over true 16QAM/64QAM bursts.

    Columns: predicted 16QAM, predicted 64QAM (other predictions are left out, so a
    row can sum to < 1).
    """
    out = np.zeros((2, 2))
    for r, t in enumerate((i16, i64)):
        m = y == t
        out[r] = [np.mean(pred[m] == i16), np.mean(pred[m] == i64)]
    return out


def qam_confusion_rate(y, pred, snr, snr_grid, i16, i64):
    """(P(16->64) + P(64->16)) / 2 per SNR."""
    rates = []
    for s in snr_grid:
        m = snr == s
        b = qam_subblock(y[m], pred[m], i16, i64)
        rates.append((b[0, 1] + b[1, 0]) / 2)
    return np.array(rates)


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test: are A and B wrong on different bursts more often than chance?

    Returns (n_a_right_b_wrong, n_a_wrong_b_right, p_value).
    """
    a_ok, b_ok = pred_a == y, pred_b == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p
