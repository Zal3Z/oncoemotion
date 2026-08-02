"""Statistics utilities (spec section 13).

Ships the stratified bootstrap CI now (pure numpy, deterministic via seed) and a
Benjamini-Hochberg multiple-comparison correction. Effect sizes and permutation
tests land with their analyses in later phases.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 12345,
    strata=None,
):
    """Bootstrap confidence interval for a statistic.

    If ``strata`` (same length as ``values``) is given, resampling is stratified
    within each stratum (spec: "bootstrap stratificato per symptom concept").
    Returns ``(point_estimate, lo, hi)``.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = float(statistic(values))

    if strata is not None:
        strata = np.asarray(strata)
        groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if strata is None:
            idx = rng.integers(0, n, size=n)
        else:
            idx = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
        boots[b] = statistic(values[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return (point, float(lo), float(hi))


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Return a boolean array of rejections under BH FDR control."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=bool)
    kmax = np.max(np.where(below)[0])
    cutoff = ranked[kmax]
    return p <= cutoff


def bh_adjusted_pvalues(pvalues):
    """BH step-up adjusted p-values (q-values), in the input order.

    Companion to :func:`benjamini_hochberg`, which returns rejections at a fixed
    alpha. Reporting the adjusted value as well lets a reader apply their own
    threshold instead of taking ours.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    out = np.empty(m, dtype=np.float64)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out
