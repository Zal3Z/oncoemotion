"""Emotion-vector datatypes + confounder orthogonalization (spec section 8).

``orthogonalize`` removes, from an emotion vector, the component lying in the
span of a set of confounder vectors (urgency, uncertainty, safety, negative
valence, ...). It uses a QR decomposition of the confounder basis — it never
materializes a (d x d) identity/projection matrix, so it scales to large hidden
sizes:

    v_perp = v - Q (Qᵀ v)     where  C = Q R  (thin QR of the confounder matrix)

which equals ``(I - P_C) v`` with ``P_C = Q Qᵀ`` the orthogonal projector onto
span(C), but costs O(d·k) instead of O(d²).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmotionVector:
    name: str                       # e.g. "afraid_alarmed"
    layer: int
    vector: np.ndarray              # shape (hidden,)
    method: str = "diff_of_means"   # diff_of_means|contrastive|pca|logistic|lda
    original_norm: float = 0.0      # norm BEFORE any normalization (spec: keep both)
    provenance: str = ""            # comprehension|generation split etc.
    meta: dict = field(default_factory=dict)

    def normalized(self) -> "EmotionVector":
        n = float(np.linalg.norm(self.vector))
        vec = self.vector / n if n > 0 else self.vector
        return EmotionVector(
            self.name, self.layer, vec, self.method,
            original_norm=self.original_norm or n, provenance=self.provenance, meta=dict(self.meta),
        )


def orthogonalize(v: np.ndarray, confounders: np.ndarray | None) -> np.ndarray:
    """Return the component of ``v`` orthogonal to span(confounders).

    Parameters
    ----------
    v : (d,) array
    confounders : (k, d) array of k confounder vectors, or None/empty.
    """
    v = np.asarray(v, dtype=np.float64)
    if confounders is None:
        return v.copy()
    C = np.asarray(confounders, dtype=np.float64)
    if C.ndim == 1:
        C = C[None, :]
    if C.size == 0 or C.shape[0] == 0:
        return v.copy()
    # Columns = confounder vectors -> shape (d, k).
    A = C.T
    # Thin QR; Q columns are an orthonormal basis of span(C).
    Q, _ = np.linalg.qr(A)
    # Remove projection onto span(C): v - Q (Qᵀ v).
    return v - Q @ (Q.T @ v)


def random_vector(dim: int, seed: int, norm: float = 1.0) -> np.ndarray:
    """Reproducible random vector with a given L2 norm (spec section 12 control).

    Deterministic for a fixed ``seed`` (uses numpy's Generator, not global state).
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    n = np.linalg.norm(v)
    return (v / n) * norm if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
