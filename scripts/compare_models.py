#!/usr/bin/env python
"""[MULTI-MODEL] Aggregate per-model reports into a cross-model comparison.

Reads outputs/models/<slug>/{vector_validation,clinical_probing,steering_effects,
patching_effects}.json and produces a comparison table (printed + JSON) and a
figure. Compares the QUALITATIVE story per model (spaces differ, so vectors are
not comparable directly):

  * afraid AUROC        — is the fear direction decodable (held-out one-vs-rest)?
  * severity trend      — mean Pearson(step, afraid-z) across gradients
  * valence confound    — Pearson(afraid-z, neg-valence-z); lower = better disentangled
  * persistence         — |afraid-z| retained through the neutral filler
  * steering vs random  — |Δentropy| emotion vs random same-norm at the severe input
  * top1 flips          — did steering ever change the decision token?

Usage:
    python scripts/compare_models.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
EMO = "afraid_alarmed"


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract(d: Path) -> dict:
    out = {"model": None}
    val = _load(d / "vector_validation.json")
    prob = _load(d / "clinical_probing.json")
    steer = _load(d / "steering_effects.json")
    patch = _load(d / "patching_effects.json")

    if val:
        out["model"] = val.get("model_id")
        c = val.get("concepts", {}).get(EMO, {})
        out["afraid_auroc"] = c.get("best_auroc")
        out["afraid_best_layer"] = c.get("best_layer")
        emo_aurocs = [v.get("best_auroc") for k, v in val.get("concepts", {}).items()
                      if v.get("kind") == "emotion" and v.get("best_auroc") is not None]
        out["mean_emotion_auroc"] = round(float(np.mean(emo_aurocs)), 3) if emo_aurocs else None
    if prob:
        out["model"] = out["model"] or prob.get("model_id")
        gt = prob.get("gradient_trends_pearson_step_vs_z", {})
        vals = [g.get(EMO) for g in gt.values() if g.get(EMO) is not None]
        out["severity_trend"] = round(float(np.mean(vals)), 3) if vals else None
        dist = prob.get("distinguishability_emotion_vs_confounder", {})
        out["valence_confound"] = dist.get(f"{EMO}~general_negative_valence")
        out["persistence"] = (prob.get("persistence_retained_fraction", {}) or {}).get(EMO)
    if steer:
        out["model"] = out["model"] or steer.get("model_id")
        sev = (steer.get("inputs", {}).get("severe", {}) or {}).get("add", {})

        def maxabs(cond):
            cur = sev.get(cond, [])
            return round(max((abs(c["delta_entropy"]) for c in cur), default=0.0), 3)

        out["steer_emotion_dH"] = maxabs("emotion")
        out["steer_random_dH"] = maxabs("random")
        flips = 0
        for tag, e in steer.get("inputs", {}).items():
            for cond, curve in e.get("add", {}).items():
                if cond == "emotion":
                    flips += sum(c.get("top1_changed", False) for c in curve)
        out["steer_top1_flips"] = flips
    if patch:
        pr = (patch.get("pairs", {}).get("severe->mild", {}) or {})
        out["patch_emotion_dH"] = (pr.get("emotion_direction", {}) or {}).get("delta_entropy")
        out["patch_random_dH"] = (pr.get("random_direction", {}) or {}).get("delta_entropy")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-root", type=Path, default=_ROOT / "outputs/models")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/reports/model_comparison.json")
    ap.add_argument("--figure", type=Path, default=_ROOT / "outputs/figures/model_comparison.png")
    ap.add_argument("--md", type=Path, default=_ROOT / "outputs/reports/model_comparison.md")
    args = ap.parse_args()

    dirs = [p for p in sorted(args.models_root.glob("*")) if p.is_dir()]
    rows = [extract(d) | {"slug": d.name} for d in dirs]
    rows = [r for r in rows if r.get("afraid_auroc") is not None or r.get("severity_trend") is not None]
    if not rows:
        print(f"No model reports found under {args.models_root}. Run scripts/run_all_models.py first.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [("slug", "model"), ("afraid_auroc", "afraidAUROC"), ("severity_trend", "sevTrend"),
            ("valence_confound", "valConf"), ("persistence", "persist"),
            ("steer_emotion_dH", "steerEmo"), ("steer_random_dH", "steerRnd"), ("steer_top1_flips", "flips")]
    print("\n=== Cross-model comparison (afraid_alarmed) ===")
    print("  " + " ".join(f"{h:>12}" for _, h in cols))
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            cells.append(f"{v:>12.3f}" if isinstance(v, float) else f"{str(v):>12}")
        print("  " + " ".join(cells))

    # markdown table
    md = ["# Cross-model comparison — emotion-like signals (afraid_alarmed)\n",
          "| model | afraid AUROC | severity trend | valence confound | persistence | steer ΔH (emo/rnd) | top1 flips |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| {slug} | {a} | {s} | {v} | {p} | {se}/{sr} | {f} |".format(
            slug=r["slug"], a=r.get("afraid_auroc"), s=r.get("severity_trend"),
            v=r.get("valence_confound"), p=r.get("persistence"),
            se=r.get("steer_emotion_dH"), sr=r.get("steer_random_dH"), f=r.get("steer_top1_flips")))
    md += ["", "Reading: higher **afraid AUROC** = fear direction more decodable. **severity trend** "
           "near +1 = fear rises with severity. **valence confound** near 0 = better disentangled from "
           "generic negative valence. **steer ΔH (emo/rnd)**: emotion effect is causal-specific only if it "
           "clearly exceeds the random-vector control. Vectors are per-model (different spaces): compare the "
           "story, not the raw numbers."]
    args.md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out} and {args.md}")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["slug"] for r in rows]
        metrics = [("afraid_auroc", "afraid AUROC"), ("severity_trend", "severity trend"),
                   ("valence_confound", "valence confound"), ("persistence", "persistence")]
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.6))
        x = np.arange(len(labels))
        for ax, (key, title) in zip(axes, metrics):
            vals = [r.get(key) if isinstance(r.get(key), (int, float)) else 0 for r in rows]
            ax.bar(x, vals, color="#4f7cc4")
            ax.axhline(0, color="gray", lw=.6)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
        # steering emotion vs random overlay on last axis? add a 5th panel
        fig.suptitle("Cross-model comparison — afraid_alarmed", fontsize=11)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=130)
        print(f"Wrote {args.figure}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
