"""Linear probing / direction evaluation (spec sections 11, 13).

Evaluate how well a concept DIRECTION separates held-out positive vs control
activations: projection AUROC, best-threshold accuracy, Cohen's d effect size,
plus a stratified bootstrap CI on the AUROC.
"""

from __future__ import annotations

import numpy as np

from oncoemotion.statistics import bootstrap_ci


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def projection_scores(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ unit(np.asarray(v, dtype=np.float64))


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def cohens_d(scores: np.ndarray, y: np.ndarray) -> float:
    a, b = scores[y == 1], scores[y == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def best_threshold_accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    s = scores[order]
    yy = y[order]
    best = 0.0
    for thr in np.concatenate([[s[0] - 1], (s[:-1] + s[1:]) / 2, [s[-1] + 1]]):
        pred = (scores >= thr).astype(int)
        acc = float((pred == y).mean())
        best = max(best, acc, 1 - acc)
    return best


def evaluate_direction(X: np.ndarray, y: np.ndarray, v: np.ndarray, seed: int = 12345,
                       n_boot: int = 500) -> dict:
    """Evaluate one direction on one layer.

    ``n_boot=0`` skips the bootstrap and returns ``auroc_ci = [None, None]``. Use it
    when sweeping layers to pick one: only the layer that gets reported needs an
    interval, and the bootstrap dominates the cost of a full sweep.
    """
    y = np.asarray(y).astype(int)
    scores = projection_scores(X, v)
    auroc = _auroc(scores, y)
    if n_boot:
        idx = np.arange(len(y))

        def _auc_stat(sample_idx):
            sample_idx = sample_idx.astype(int)
            return _auroc(scores[sample_idx], y[sample_idx])

        point, lo, hi = bootstrap_ci(idx, statistic=_auc_stat, n_boot=n_boot, seed=seed, strata=y)
    else:
        lo = hi = None
    return {
        "auroc": auroc,
        "auroc_ci": [lo, hi],
        "accuracy": best_threshold_accuracy(scores, y),
        "cohens_d": cohens_d(scores, y),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }
