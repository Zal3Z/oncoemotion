"""Construct emotion concept vectors per layer (spec sections 7-8).

Given pooled activations ``X`` (shape [n, hidden]) and binary labels at one
layer, build a concept direction with several methods:

  * diff_of_means   : mean(pos) - mean(neg)
  * pca             : first principal component, oriented toward diff_of_means
  * logistic        : logistic-regression weight direction
  * lda             : linear-discriminant direction (regularized)

Two backends compute the SAME directions:
  * CPU (numpy + scikit-learn) — default, dependency-light, used in tests;
  * GPU (torch) — pass ``device="cuda"``; the covariance / solve / SVD run on the
    GPU, so all four methods stay fast even at large hidden sizes (e.g. 4096),
    where the CPU LDA/logistic are the bottleneck.

The ORIGINAL norm is stored before normalization. Confounder orthogonalization is
applied via :func:`oncoemotion.emotion_vectors.orthogonalize`.
"""

from __future__ import annotations

import numpy as np

from oncoemotion.emotion_vectors.vectors import EmotionVector, orthogonalize

METHODS = ("diff_of_means", "pca", "logistic", "lda")
LDA_SHRINKAGE = 0.5  # shrink toward scaled identity (stable when hidden >> n)


def _orient(v: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return -v if float(np.dot(v, reference)) < 0 else v


def _use_torch(device) -> bool:
    if not device or str(device) == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _direction_sklearn(X, y, method, dom):
    if method == "diff_of_means":
        return dom
    if method == "pca":
        from sklearn.decomposition import PCA

        p = PCA(n_components=1)
        p.fit(X - X.mean(0))
        return _orient(p.components_[0], dom)
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X, y)
        return _orient(clf.coef_[0], dom)
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X, y)
        return _orient(lda.coef_[0], dom)
    raise ValueError(f"Unknown method: {method}")


def _direction_torch(X, y, method, dom, device):
    """Same directions as the sklearn backend, computed on the GPU."""
    import torch

    dev = torch.device(device)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev)
    yt = torch.as_tensor(np.asarray(y).astype(np.int64), device=dev)
    domt = torch.as_tensor(np.asarray(dom, dtype=np.float32), device=dev)
    pos, neg = Xt[yt == 1], Xt[yt == 0]

    def orient(w):
        return w if torch.dot(w, domt) >= 0 else -w

    if method == "diff_of_means":
        v = domt
    elif method == "pca":
        Xc = Xt - Xt.mean(0)
        try:
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            v = orient(Vh[0])
        except Exception:
            v = domt
    elif method == "logistic":
        H = Xt.shape[1]
        lin = torch.nn.Linear(H, 1).to(dev)
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
        npos = (yt == 1).sum().clamp(min=1).float()
        pw = (yt == 0).sum().float() / npos
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-3)
        yf = yt.float()
        for _ in range(300):
            opt.zero_grad()
            loss = lossf(lin(Xt).squeeze(-1), yf)
            loss.backward()
            opt.step()
        v = orient(lin.weight.detach()[0])
    elif method == "lda":
        Xc = torch.cat([pos - pos.mean(0), neg - neg.mean(0)], 0)
        n, H = Xc.shape
        Sw = (Xc.t() @ Xc) / max(n - 2, 1)
        eye = torch.eye(H, device=dev, dtype=Sw.dtype)
        Sw = (1.0 - LDA_SHRINKAGE) * Sw + LDA_SHRINKAGE * (torch.trace(Sw) / H) * eye
        v = orient(torch.linalg.solve(Sw, domt))
    else:
        raise ValueError(f"Unknown method: {method}")
    return v.detach().to(torch.float64).cpu().numpy()


def build_layer_vector(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    concept: str,
    layer: int,
    confounders: np.ndarray | None = None,
    device=None,
) -> EmotionVector:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pos, neg = X[y == 1], X[y == 0]
    dom = pos.mean(0) - neg.mean(0)  # reference direction

    if _use_torch(device):
        v = _direction_torch(X, y, method, dom, device)
    else:
        v = _direction_sklearn(X, y, method, dom)

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
    device=None,
) -> list[EmotionVector]:
    """Build a vector for every layer. ``acts`` shape [n, L+1, H]."""
    n_layers = acts.shape[1]
    out = []
    for l in range(n_layers):
        conf = confounders_per_layer[l] if confounders_per_layer else None
        out.append(build_layer_vector(acts[:, l, :], y, method, concept, l, conf, device=device))
    return out
