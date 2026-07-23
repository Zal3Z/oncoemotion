"""Steering/ablation specs and the underlying vector ops (array-generic).

The math is written against numpy arrays so it is unit-tested without torch; the
Phase-4 runtime wraps these in forward hooks operating on torch tensors (the
same elementwise formulas apply).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SteeringSpec:
    vector_name: str
    layers: list[int]
    alpha: float
    target: str = "residual"          # residual|mlp|attn
    token_scope: str = "all_post"     # all_post|pre_decision
    note: str = ""


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def steer_add(h: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """h' = h + alpha * v  (v broadcast over the token/batch axes)."""
    return np.asarray(h, dtype=np.float64) + float(alpha) * np.asarray(v, dtype=np.float64)


def ablate_projection(h: np.ndarray, v: np.ndarray) -> np.ndarray:
    """h' = h - proj_v(h)  — remove the component of h along v."""
    h = np.asarray(h, dtype=np.float64)
    vhat = _unit(np.asarray(v, dtype=np.float64))
    # coefficient per row = h·vhat ; subtract along vhat.
    coeff = h @ vhat
    return h - np.outer(coeff, vhat) if h.ndim > 1 else h - coeff * vhat


def norm_scaled_alpha(base_alpha: float, residual_norm: float, ref_norm: float = 1.0) -> float:
    """Adapt alpha to the residual-stream norm (spec section 12)."""
    if ref_norm <= 0:
        return base_alpha
    return base_alpha * (residual_norm / ref_norm)
