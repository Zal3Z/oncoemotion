#!/usr/bin/env python
"""[PHASE 2] Validate emotion vectors on held-out text (spec sections 7, 13).

One-vs-rest held-out evaluation: for each concept and layer, project the test
split and measure AUROC (concept vs all other test items) with bootstrap CI,
best-threshold accuracy, Cohen's d. Emotions use the RESIDUALIZED vectors (conf-
ounder-orthogonalized) when available; controls use raw. Reports the best layer
per concept, the layer sweep, and cross-concept collinearity.

Usage:
    python scripts/validate_vectors.py                 # emotions: residualized
    python scripts/validate_vectors.py --variant raw   # force raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.probing.probe import evaluate_direction  # noqa: E402
from oncoemotion.emotion_vectors.vectors import cosine  # noqa: E402
from oncoemotion.emotion_vectors.seeds import CONTROL_SEEDS, EMOTION_SEEDS  # noqa: E402

EMOTIONS = set(EMOTION_SEEDS)
CONTROLS = set(CONTROL_SEEDS)


def vec_key(V, concept, method, variant):
    """Residualized for emotions when available, else raw."""
    if concept in EMOTIONS and variant != "raw" and f"{concept}|{method}|resid" in V:
        return f"{concept}|{method}|resid"
    return f"{concept}|{method}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", type=Path, default=_ROOT / "outputs/activations/emotion_acts.npz")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--figure", type=Path, default=_ROOT / "outputs/figures/layer_sweep_auroc.png")
    args = ap.parse_args()

    A = np.load(args.acts, allow_pickle=True)
    acts, concepts, splits = A["acts"], A["concepts"], A["splits"]
    V = np.load(args.vecs, allow_pickle=True)
    n_layers = acts.shape[1]
    test = splits == "test"

    report = {"model_id": str(A["model_id"]), "method": args.method, "variant": args.variant,
              "n_layers": n_layers, "eval": "one_vs_rest_heldout", "concepts": {}}
    sweep, best_vec = {}, {}
    all_concepts = sorted(set(concepts.tolist()) - {"neutral"})

    for concept in all_concepts:
        key = vec_key(V, concept, args.method, args.variant)
        if key not in V:
            continue
        layer_vecs = V[key]
        pos = test & (concepts == concept)
        neg = test & (concepts != concept)   # one-vs-rest (neutral + others)
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        idx = np.where(pos | neg)[0]
        y = (concepts[idx] == concept).astype(int)
        per_layer, aurocs = [], []
        for l in range(n_layers):
            res = evaluate_direction(acts[idx, l, :], y, layer_vecs[l])
            per_layer.append({"layer": l, **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                             for k, v in res.items()}})
            aurocs.append(res["auroc"] if not np.isnan(res["auroc"]) else 0.0)
        best_l = int(np.argmax([p["auroc"] if p["auroc"] is not None else -1 for p in per_layer]))
        sweep[concept] = aurocs
        best_vec[concept] = layer_vecs[best_l]
        report["concepts"][concept] = {
            "kind": "emotion" if concept in EMOTIONS else "control",
            "vector_key": key,
            "best_layer": best_l,
            "best_auroc": per_layer[best_l]["auroc"],
            "best_auroc_ci": per_layer[best_l]["auroc_ci"],
            "best_cohens_d": per_layer[best_l]["cohens_d"],
            "layer_sweep": per_layer,
        }

    names = sorted(best_vec)
    report["collinearity_best_layer"] = {
        f"{a}~{b}": round(cosine(best_vec[a], best_vec[b]), 4)
        for a in names for b in names if a < b
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report -> {args.report}")

    print(f"\n{'concept':24} {'kind':8} {'best_L':>6} {'AUROC':>7} {'CI':>16} {'d':>6}")
    for c in sorted(report["concepts"]):
        r = report["concepts"][c]
        ci = r["best_auroc_ci"]
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci and ci[0] is not None else "-"
        au = r["best_auroc"] or 0
        print(f"{c:24} {r['kind']:8} {r['best_layer']:>6} {au:>7.3f} {ci_s:>16} {r['best_cohens_d']:>6.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for concept, aurocs in sweep.items():
            style = "-" if concept in EMOTIONS else "--"
            ax.plot(range(n_layers), aurocs, style, label=concept, alpha=0.8)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("layer"); ax.set_ylabel("held-out AUROC (one-vs-rest)")
        ax.set_title(f"Emotion/control direction — layer sweep ({args.method}, {args.variant})")
        ax.legend(fontsize=7, ncol=2)
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.figure, dpi=130)
        print(f"Wrote figure -> {args.figure}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
