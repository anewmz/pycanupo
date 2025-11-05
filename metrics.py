"""
Utility functions for evaluating classification results.
"""

import numpy as np


def _safe_metrics(y_ref: np.ndarray, y_pred: np.ndarray):
    """Compute metrics comparing reference (y_ref) to predictions (y_pred).
    y_ref: integers {0,1}; y_pred: {-1,0,1}, with -1 meaning "rejected"
    Returns dict with accuracy, confusion matrix, precision, recall, F1, IoU, kappa."""
    # y_ref: integers {0,1}; y_pred: {-1,0,1}, with -1 meaning "rejected"
    mask = y_pred != -1
    if mask.sum() == 0:
        return {"acc": 0.0, "cm": np.zeros((2,2), dtype=int), "prec": [0,0], "rec": [0,0], "f1": [0,0], "iou": [0,0], "kappa": 0.0, "used_pts": 0}
    a = y_ref[mask].astype(int)
    b = y_pred[mask].astype(int)

    # confusion
    cm = np.zeros((2,2), dtype=int)
    for aa, bb in zip(a, b):
        cm[aa, bb] += 1
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp+fp>0, tp/(tp+fp), 0.0)
        rec  = np.where(tp+fn>0, tp/(tp+fn), 0.0)
        f1   = np.where(prec+rec>0, 2*prec*rec/(prec+rec), 0.0)
        iou  = np.where(tp+fp+fn>0, tp/(tp+fp+fn), 0.0)

    acc = (tp.sum()+tn.sum())/cm.sum() if cm.sum()>0 else 0.0

    # Cohen's kappa
    p0 = acc
    pe = ((cm.sum(axis=0)/cm.sum()) * (cm.sum(axis=1)/cm.sum())).sum() if cm.sum()>0 else 0.0
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    return {
        "acc": float(acc),
        "cm": cm,
        "prec": [float(prec[0]), float(prec[1])],
        "rec":  [float(rec[0]),  float(rec[1])],
        "f1":   [float(f1[0]),   float(f1[1])],
        "iou":  [float(iou[0]),  float(iou[1])],
        "kappa": float(kappa),
        "used_pts": int(mask.sum())
    }
