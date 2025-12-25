from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def oos_auc(y_is_oos: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_is_oos, score))


def fpr_at_95tpr(y_is_oos: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_is_oos, score)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])
