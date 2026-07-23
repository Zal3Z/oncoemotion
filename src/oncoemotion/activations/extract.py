"""Pooled hidden-state extraction (spec sections 6, 10).

Runs each text through a loaded :class:`ModelAdapter` with
``output_hidden_states=True`` and pools per layer. Returns a numpy array of shape
``[n_texts, n_layers+1, hidden]`` (layer 0 is the embedding output).

Pooling modes: ``mean`` (over real tokens) and ``last`` (last real token) —
these correspond to the mean-pooling and last-token-pooling of spec section 10.
Deterministic: no sampling; identical across batch sizes (processed one text at a
time here for clarity and exactness).
"""

from __future__ import annotations

import numpy as np


def pooled_hidden_states(adapter, texts, pooling: str = "mean", progress: bool = False) -> np.ndarray:
    import torch

    feats: list[np.ndarray] = []
    n = len(texts)
    for i, text in enumerate(texts):
        cap = adapter.forward_capture(text)
        hs = cap["hidden_states"]  # tuple (L+1) of [1, seq, H]
        attn = cap.get("attention_mask")
        layer_vecs = []
        for h in hs:
            h0 = h[0]  # [seq, H]
            if pooling == "last":
                vec = h0[-1]
            else:  # mean over real tokens
                if attn is not None:
                    m = attn[0].to(h0.dtype).unsqueeze(-1)  # [seq,1]
                    vec = (h0 * m).sum(0) / m.sum().clamp(min=1)
                else:
                    vec = h0.mean(0)
            layer_vecs.append(vec.float().cpu().numpy())
        feats.append(np.stack(layer_vecs, axis=0))  # [L+1, H]
        if progress and (i % 20 == 0 or i == n - 1):
            print(f"  extracted {i + 1}/{n}", flush=True)
    return np.stack(feats, axis=0)  # [n, L+1, H]
