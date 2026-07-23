"""Construct emotion concept vectors per layer (spec sections 7-8).

Given pooled activations ``X`` (shape [n, hidden]) and binary labels at one
layer, build a concept direction with several methods:

  * diff_of_means   : mean(pos) - mean(neg)
  * pca             : first principal component, oriented toward diff_of_means
  * logistic        : logistic-regression weight direction
  * lda             : linear-discriminant direction

The ORIGINAL norm is stored before normalization (spec: keep both). Confounder
orthogonalization is applied via :func:`oncoemotion.emotion_vectors.orthogonalize`.
"""

from __future__ import annotations

import numpy as np

from oncoemotion.emotion_vectors.vectors import EmotionVector, orthogonalize

METHODS = ("diff_of_means", "pca", "logistic", "lda")


def _orient(v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip v to point in the same half-space as the reference direction."""
    return -v if float(np.dot(v, reference)) < 0 else v


def build_layer_vector(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    concept: str,
    layer: int,
    confounders: np.ndarray | None = None,
) -> EmotionVector:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pos, neg = X[y == 1], X[y == 0]
    dom = pos.mean(0) - neg.mean(0)  # reference direction

    if method == "diff_of_means":
        v = dom
    elif method == "pca":
        from sklearn.decomposition import PCA

        p = PCA(n_components=1)
        p.fit(X - X.mean(0))
        v = _orient(p.components_[0], dom)
    elif method == "logistic":
        from sklearn.linear_model import LogisticRegression

        # class_weight balanced: negatives (one-vs-rest) far outnumber positives.
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X, y)
        v = _orient(clf.coef_[0], dom)
    elif method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X, y)
        v = _orient(lda.coef_[0], dom)
    else:
        raise ValueError(f"Unknown method: {method}")

    if confounders is not None:
        v = orthogonalize(v, confounders)

    return EmotionVector(
        name=concept,
        layer=layer,
        vector=v,
        method=method,
        original_norm=float(np.linalg.norm(v)),
        provenance="comprehension",
    )


def build_all_layers(
    acts: np.ndarray,
    y: np.ndarray,
    concept: str,
    method: str = "diff_of_means",
    confounders_per_layer: list[np.ndarray] | None = None,
) -> list[EmotionVector]:
    """Build a vector for every layer. ``acts`` shape [n, L+1, H]."""
    n_layers = acts.shape[1]
    out = []
    for l in range(n_layers):
        conf = confounders_per_layer[l] if confounders_per_layer else None
        out.append(build_layer_vector(acts[:, l, :], y, method, concept, l, conf))
    return out
