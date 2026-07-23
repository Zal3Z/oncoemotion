#!/usr/bin/env python
"""[PHASE 3] Measure emotion-like signals during the PRO-CTCAE decision (sections 10-11).

For each controlled clinical input:
  1. build the decision prompt with the identical teacher-forced prefix
     `{"pro_ctcae":{"term":"` ; point E = the last token (pre-decision);
  2. capture point-E hidden states and project onto the Phase-2 emotion/control
     vectors (at each concept's best validation layer);
  3. z-score against a neutral baseline.

Analyses (RQ1-RQ6): severity-gradient trends, emotion-vs-confounder
distinguishability, and persistence (re-measure with an identical neutral filler
inserted between the patient text and the decision).

Usage:
    python scripts/run_probing.py
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
from oncoemotion.clinical.prompt import build_decision_prompt, NEUTRAL_FILLER  # noqa: E402
from oncoemotion.clinical.measure import point_e_hidden, project_scores, zscore, decision_summary  # noqa: E402
from oncoemotion.clinical.measure_dataset import build_measure_items  # noqa: E402

EMOTIONS = ["afraid_alarmed", "anxious_nervous", "sad", "calm", "compassionate"]
CONFOUNDERS = ["clinical_severity", "urgency", "safety_policy", "general_negative_valence"]


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--report", type=Path, default=_ROOT / "outputs/reports/clinical_probing.json")
    ap.add_argument("--figure", type=Path, default=_ROOT / "outputs/figures/clinical_gradients.png")
    args = ap.parse_args()

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    best_layer = {c: val["concepts"][c]["best_layer"] for c in val["concepts"]}

    def key_for(c):
        rk = f"{c}|{args.method}|resid"
        if c in EMOTIONS and args.variant != "raw" and rk in V:
            return rk
        return f"{c}|{args.method}"

    concepts = [c for c in (EMOTIONS + CONFOUNDERS) if key_for(c) in V]
    vectors = {c: V[key_for(c)] for c in concepts}
    layer_of = {c: best_layer.get(c, vectors[c].shape[0] // 2) for c in concepts}
    print("vector variant:", {c: ("resid" if key_for(c).endswith("resid") else "raw") for c in concepts})

    cfg = ModelConfig(dtype=args.dtype, device_map=args.device)
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()

    items = build_measure_items()
    print(f"Measuring {len(items)} clinical inputs at point E ...", flush=True)

    # 1) raw projections at point E (no filler)
    raw = {}
    summaries = {}
    for it in items:
        prompt = build_decision_prompt(it.text)
        h = point_e_hidden(adapter, prompt)
        raw[it.item_id] = project_scores(h, vectors, layer_of)
        summaries[it.item_id] = decision_summary(adapter, prompt)

    # 2) neutral baseline -> z-scores
    neutral_ids = [it.item_id for it in items if it.is_neutral]
    base_mean = {c: float(np.mean([raw[i][c] for i in neutral_ids])) for c in concepts}
    base_std = {c: float(np.std([raw[i][c] for i in neutral_ids]) + 1e-9) for c in concepts}
    z = {iid: zscore(sc, base_mean, base_std) for iid, sc in raw.items()}

    by_id = {it.item_id: it for it in items}

    # 3) severity-gradient trends: Pearson(step, z) per concept per gradient
    gradients = {}
    for it in items:
        if it.group.startswith("gradient:"):
            gradients.setdefault(it.group, []).append(it)
    grad_trends = {}
    for g, its in gradients.items():
        its = sorted(its, key=lambda x: x.step)
        steps = [x.step for x in its]
        grad_trends[g] = {c: round(_pearson(steps, [z[x.item_id][c] for x in its]), 3) for c in concepts}

    # 4) distinguishability: correlation emotion-z vs confounder-z across gradient items
    grad_item_ids = [it.item_id for it in items if it.group.startswith("gradient:")]
    distinguish = {}
    for e in [c for c in EMOTIONS if c in concepts]:
        for cf in [c for c in CONFOUNDERS if c in concepts]:
            distinguish[f"{e}~{cf}"] = round(
                _pearson([z[i][e] for i in grad_item_ids], [z[i][cf] for i in grad_item_ids]), 3)

    # 5) persistence: re-measure gradient items WITH the neutral filler
    persistence = {}
    for it in items:
        if not it.group.startswith("gradient:"):
            continue
        h_f = point_e_hidden(adapter, build_decision_prompt(it.text, neutral_filler=NEUTRAL_FILLER))
        zf = zscore(project_scores(h_f, vectors, layer_of), base_mean, base_std)
        persistence[it.item_id] = {c: round(zf[c], 3) for c in concepts}
    # retained fraction for the key emotions (|z_with_filler| / |z_without|)
    retain = {}
    for c in [x for x in EMOTIONS if x in concepts]:
        num, den = 0.0, 0.0
        for iid in persistence:
            num += abs(persistence[iid][c]); den += abs(z[iid][c])
        retain[c] = round(num / den, 3) if den > 1e-9 else None

    report = {
        "model_id": adapter.config.model_id,
        "method": args.method,
        "layer_of": layer_of,
        "gradient_trends_pearson_step_vs_z": grad_trends,
        "distinguishability_emotion_vs_confounder": distinguish,
        "persistence_retained_fraction": retain,
        "per_item": {
            iid: {
                "text": by_id[iid].text, "group": by_id[iid].group, "step": by_id[iid].step,
                "formulation": by_id[iid].formulation, "is_neutral": by_id[iid].is_neutral,
                "z": {c: round(z[iid][c], 3) for c in concepts},
                "entropy": round(summaries[iid]["entropy"], 3),
                "margin": round(summaries[iid]["top1_top2_margin"], 4),
            } for iid in raw
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report -> {args.report}")

    # print gradient trend summary
    print("\nSeverity-gradient trend (Pearson step vs z-score):")
    print(f"  {'gradient':22} " + " ".join(f"{c[:10]:>10}" for c in concepts))
    for g, tr in grad_trends.items():
        print(f"  {g:22} " + " ".join(f"{tr[c]:>10.2f}" for c in concepts))
    print("\nPersistence (|z| retained through neutral filler):", retain)

    # figure: emotion & confounder z vs severity step, per gradient
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gl = sorted(gradients)
        fig, axes = plt.subplots(1, len(gl), figsize=(3.4 * len(gl), 4), sharey=True)
        if len(gl) == 1:
            axes = [axes]
        show = [c for c in ["afraid_alarmed", "anxious_nervous", "sad", "calm",
                            "clinical_severity", "urgency"] if c in concepts]
        for ax, g in zip(axes, gl):
            its = sorted(gradients[g], key=lambda x: x.step)
            steps = [x.step for x in its]
            for c in show:
                ls = "-" if c in EMOTIONS else "--"
                ax.plot(steps, [z[x.item_id][c] for x in its], ls, marker="o", ms=3, label=c, alpha=0.85)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.set_title(g.replace("gradient:", ""), fontsize=9)
            ax.set_xlabel("severity step")
        axes[0].set_ylabel("z-score vs neutral (point E)")
        axes[-1].legend(fontsize=6, ncol=1, loc="upper left")
        fig.suptitle("Emotion-like signal at point E vs symptom severity")
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.figure, dpi=130)
        print(f"Wrote figure -> {args.figure}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
