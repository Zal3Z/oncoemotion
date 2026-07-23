"""Capture point-E activations + decision summary, and project onto vectors.

All functions are deterministic. Projection uses the unit vector at a chosen
layer; z-scores are computed against a neutral baseline (spec section 11).
"""

from __future__ import annotations

import numpy as np


def point_e_hidden(adapter, prompt: str) -> np.ndarray:
    """Hidden states at the LAST token (point E). Returns [L+1, H]."""
    cap = adapter.forward_capture(prompt)
    hs = cap["hidden_states"]
    return np.stack([h[0, -1].float().cpu().numpy() for h in hs], axis=0)


def token_by_token_hidden(adapter, prompt: str, layer: int) -> np.ndarray:
    """Hidden states at every token for one layer. Returns [seq, H]."""
    cap = adapter.forward_capture(prompt)
    return cap["hidden_states"][layer][0].float().cpu().numpy()


def decision_summary(adapter, prompt: str) -> dict:
    """Next-token distribution summary at point E (entropy, top1-top2 margin)."""
    import torch

    cap = adapter.forward_capture(prompt)
    logits = cap["logits"][0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    topv, topi = torch.topk(probs, 5)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
    margin = float(topv[0] - topv[1])
    return {
        "entropy": entropy,
        "top1_top2_margin": margin,
        "top1_prob": float(topv[0]),
        "top_token_ids": [int(i) for i in topi.tolist()],
    }


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def project_scores(hidden_LH: np.ndarray, vectors: dict[str, np.ndarray],
                   layer_of: dict[str, int]) -> dict[str, float]:
    """Projection of point-E hidden state onto each concept's direction.

    ``vectors[concept]`` has shape [L+1, H]; ``layer_of[concept]`` selects the
    layer (e.g. the best validation layer). Returns concept -> scalar score.
    """
    out = {}
    for concept, vec_LH in vectors.items():
        l = layer_of.get(concept, vec_LH.shape[0] // 2)
        out[concept] = float(hidden_LH[l] @ _unit(vec_LH[l]))
    return out


def zscore(scores: dict[str, float], baseline_mean: dict[str, float],
           baseline_std: dict[str, float]) -> dict[str, float]:
    out = {}
    for c, s in scores.items():
        sd = baseline_std.get(c, 0.0)
        out[c] = float((s - baseline_mean.get(c, 0.0)) / sd) if sd > 1e-9 else 0.0
    return out
