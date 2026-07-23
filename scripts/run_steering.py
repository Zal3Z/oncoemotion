#!/usr/bin/env python
"""[PHASE 4] Causal activation-steering experiment at the decision point (section 12).

Applies transient forward-hook interventions at a chosen layer while the model is
at point E (pre-decision) and measures the causal effect on the next-token
decision (entropy, top1-top2 margin, top-1 change). Conditions (spec section 12):
baseline, +emotion, -emotion, +opposite emotion, +confounder (urgency),
+random same-norm, and ablation. Alpha grid is norm-scaled to the residual stream.

No weights are modified (hooks removed after each run).

Usage:
    python scripts/run_steering.py --concept afraid_alarmed --opposite calm --confounder urgency
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.steering.runtime import SteeringRuntime  # noqa: E402
from oncoemotion.clinical.prompt import build_decision_prompt  # noqa: E402
from oncoemotion.emotion_vectors.vectors import random_vector  # noqa: E402

ALPHA_GRID = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]

# a few probe inputs spanning severity
INPUTS = [
    ("mild", "ho un leggero fastidio ogni tanto"),
    ("severe", "ho un dolore lancinante e insopportabile che non passa"),
    ("neutral_symptom", "ho la nausea"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--concept", default="afraid_alarmed")
    ap.add_argument("--opposite", default="calm")
    ap.add_argument("--confounder", default="urgency")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = best validation layer")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--report", type=Path, default=_ROOT / "outputs/reports/steering_effects.json")
    ap.add_argument("--figure", type=Path, default=_ROOT / "outputs/figures/steering_effects.png")
    args = ap.parse_args()

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    layer = args.layer if args.layer >= 0 else val["concepts"][args.concept]["best_layer"]
    emotions = set(val.get("concepts", {}).keys()) & {
        "afraid_alarmed", "anxious_nervous", "calm", "surprised", "confused",
        "frustrated", "compassionate", "sad", "concerned"}

    def vec(concept):
        rk = f"{concept}|{args.method}|resid"
        if concept in emotions and args.variant != "raw" and rk in V:
            return V[rk][layer]
        return V[f"{concept}|{args.method}"][layer]

    v_emotion = vec(args.concept)
    v_opposite = vec(args.opposite)
    v_confounder = vec(args.confounder)
    v_random = random_vector(v_emotion.shape[0], seed=777, norm=float(np.linalg.norm(v_emotion)))

    cfg = ModelConfig(dtype=args.dtype, device_map=args.device)
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()
    rt = SteeringRuntime(adapter)
    print(f"Steering at layer {layer} (concept={args.concept})", flush=True)

    results = {"model_id": adapter.config.model_id, "layer": int(layer), "concept": args.concept,
               "alpha_grid": ALPHA_GRID, "inputs": {}}

    add_conditions = {
        "emotion": v_emotion,
        "opposite": v_opposite,
        "confounder": v_confounder,
        "random": v_random,
    }

    for tag, text in INPUTS:
        prompt = build_decision_prompt(text)
        entry = {"text": text, "add": {}, "ablation": None}
        for cond, v in add_conditions.items():
            curve = []
            for a in ALPHA_GRID:
                r = rt.steer_and_summarize(prompt, layer, v, a, mode="add")
                curve.append({"alpha": a, "delta_entropy": round(r["delta_entropy"], 4),
                              "delta_margin": round(r["delta_margin"], 4),
                              "top1_changed": r["top1_changed"]})
            entry["add"][cond] = curve
        abl = rt.steer_and_summarize(prompt, layer, v_emotion, 0.0, mode="ablate")
        entry["ablation"] = {"delta_entropy": round(abl["delta_entropy"], 4),
                             "delta_margin": round(abl["delta_margin"], 4),
                             "top1_changed": abl["top1_changed"]}
        results["inputs"][tag] = entry
        print(f"  {tag}: ablation dEntropy={abl['delta_entropy']:+.3f} top1_changed={abl['top1_changed']}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report -> {args.report}")

    # figure: delta entropy vs alpha per condition (severe input)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = list(results["inputs"])
        fig, axes = plt.subplots(1, len(tags), figsize=(4 * len(tags), 4), sharey=True)
        if len(tags) == 1:
            axes = [axes]
        for ax, tag in zip(axes, tags):
            for cond, curve in results["inputs"][tag]["add"].items():
                ax.plot([c["alpha"] for c in curve], [c["delta_entropy"] for c in curve],
                        marker="o", ms=3, label=cond)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.axvline(0, color="gray", lw=0.6, ls=":")
            ax.set_title(tag, fontsize=9); ax.set_xlabel("alpha (norm-scaled)")
        axes[0].set_ylabel("Δ entropy at point E")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"Causal steering effect — {args.concept} @ layer {layer}")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.figure, dpi=130)
        print(f"Wrote figure -> {args.figure}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
