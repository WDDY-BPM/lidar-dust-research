"""Dust is positive; false deletion is FP / (FP + TN)."""

import numpy as np

from dust.data import IGNORE


def confusion(target, probability, threshold=.5):
    if target.shape != probability.shape:
        raise ValueError("Targets and probabilities must align")
    selected = target != IGNORE
    truth, score = target[selected], probability[selected]
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("Invalid supervised targets or missing predictions")
    if np.any((score < 0) | (score > 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    pred = score >= threshold
    return np.array([np.sum((truth == 0) & ~pred), np.sum((truth == 0) & pred),
                     np.sum((truth == 1) & ~pred), np.sum((truth == 1) & pred)], dtype=np.int64)


def summarize(counts):
    tn, fp, fn, tp = map(int, counts)
    def ratio(a, b):
        return a / b if b else None
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "n_evaluated": tn + fp + fn + tp,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "dust_iou": ratio(tp, tp + fp + fn),
            "false_deletion_rate": ratio(fp, fp + tn)}
