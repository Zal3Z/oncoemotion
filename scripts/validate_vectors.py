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

from oncoemotion.probing.probe import evaluate_direction, projection_scores  # noqa: E402
from oncoemotion.emotion_vectors.vectors import cosine, orthogonalize  # noqa: E402
from oncoemotion.emotion_vectors.build import build_layer_vector  # noqa: E402
from oncoemotion.emotion_vectors.seeds import (  # noqa: E402
    CONTROL_SEEDS, EMOTION_SEEDS, LEXICAL_CONTROLS)

EMOTIONS = set(EMOTION_SEEDS)
CONTROLS = set(CONTROL_SEEDS)
# see build_vectors.py: the lexical controls are measured against the emotion axes,
# never residualized out of them
CONFOUNDER_BASIS = sorted(CONTROLS - set(LEXICAL_CONTROLS))


def vec_key(V, concept, method, variant):
    """Residualized for emotions when available, else raw."""
    if concept in EMOTIONS and variant != "raw" and f"{concept}|{method}|resid" in V:
        return f"{concept}|{method}|resid"
    return f"{concept}|{method}"


def _stratified_folds(concepts_arr: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each example to one of k folds, balanced within concept."""
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(concepts_arr), dtype=int)
    for c in sorted(set(concepts_arr.tolist())):
        idx = np.where(concepts_arr == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cross_validated_scores(acts, concepts_arr, all_concepts, band, k, seed, method, variant):
    """Out-of-fold projection scores for every concept at every layer in the band.

    Rebuilds the directions inside each fold, so no example is ever projected on a
    direction estimated from itself. Every example ends up in exactly one held-out
    fold, which is what lifts the evaluation from 2-3 positives per concept (a
    single 20% test split) to all of them.
    """
    fold = _stratified_folds(concepts_arr, k, seed)
    n = len(concepts_arr)
    scores = {c: np.full((len(band), n), np.nan) for c in all_concepts}
    for f in range(k):
        tr, te = fold != f, fold == f
        te_idx = np.where(te)[0]
        for li, l in enumerate(band):
            Xl = acts[:, l, :]
            # control directions first: they are the confounder basis
            ctrl = {}
            for c in CONFOUNDER_BASIS:
                pos = tr & (concepts_arr == c)
                neg = tr & (concepts_arr != c)
                if pos.sum() == 0 or neg.sum() == 0:
                    continue
                y = np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                X = np.concatenate([Xl[pos], Xl[neg]], axis=0)
                ctrl[c] = build_layer_vector(X, y, "diff_of_means", c, l).vector
            C = np.stack(list(ctrl.values())) if ctrl else None
            for c in all_concepts:
                pos = tr & (concepts_arr == c)
                neg = tr & (concepts_arr != c)
                if pos.sum() == 0 or neg.sum() == 0:
                    continue
                y = np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
                X = np.concatenate([Xl[pos], Xl[neg]], axis=0)
                v = build_layer_vector(X, y, method, c, l).vector
                if c in EMOTIONS and variant != "raw" and C is not None:
                    v = orthogonalize(v, C)
                scores[c][li, te_idx] = projection_scores(Xl[te_idx], v)
    return scores


def _lexical_gate(V, args, layer: int) -> dict:
    """cos(emotion axis, lexical-control axis) at the layer actually used downstream.

    The ontological gate. An emotion direction has to encode a state, not the
    presence of the word naming it. ``emotion_word_mention`` holds sentences where
    the emotion word is quoted, defined or counted; ``emotion_negated`` holds
    sentences where it is explicitly denied. Both are saturated with emotion
    vocabulary and carry no emotional state, so a high cosine means the axis is a
    word detector and the study's first step does not hold.
    """
    out = {"layer": layer, "per_emotion": {}, "controls_used": []}
    lex = {}
    for c in LEXICAL_CONTROLS:
        key = f"{c}|{args.method}"
        if key in V:
            lex[c] = V[key][layer]
    out["controls_used"] = sorted(lex)
    if not lex:
        out["note"] = ("lexical control directions absent: rebuild vectors after adding "
                       "emotion_word_mention / emotion_negated to seeds.py")
        return out
    for c in sorted(EMOTIONS):
        key = vec_key(V, c, args.method, args.variant)
        if key not in V:
            continue
        v = V[key][layer]
        out["per_emotion"][c] = {name: round(cosine(v, lv), 4) for name, lv in lex.items()}
    if out["per_emotion"]:
        allv = [abs(x) for d in out["per_emotion"].values() for x in d.values()]
        out["max_abs_cos"] = round(max(allv), 4)
        out["median_abs_cos"] = round(float(np.median(allv)), 4)
        out["fear"] = out["per_emotion"].get("afraid_alarmed")
    return out


def _print_lexical_gate(g: dict) -> None:
    print(f"\n=== CANCELLO LESSICALE (layer {g['layer']}) ===")
    if not g.get("per_emotion"):
        print(f"  {g.get('note', 'non calcolabile')}")
        return
    print(f"  cos con gli assi lessicali: mediana |cos| {g['median_abs_cos']}, "
          f"massimo {g['max_abs_cos']}")
    if g.get("fear"):
        print(f"  asse paura: " + "  ".join(f"{k}={v:+.3f}" for k, v in g["fear"].items()))
    worst = sorted(g["per_emotion"].items(),
                   key=lambda kv: -max(abs(x) for x in kv[1].values()))[:3]
    for name, d in worst:
        print(f"  piu lessicale: {name:20s} " + "  ".join(f"{k}={v:+.3f}" for k, v in d.items()))
    if g["max_abs_cos"] > 0.5:
        print("  [!] un asse supera 0.5 di allineamento col lessico: e un rilevatore di parole.")


def _run_cv(args, acts, concepts, all_concepts, band, n_layers, V, report) -> int:
    """Cross-validated evaluation with ONE layer shared by every concept.

    Two changes from the per-concept path, both aimed at the same failure. Picking
    ``argmax AUROC`` per concept over every layer, from 2-3 held-out positives, is a
    maximum over dozens of correlated noisy estimates: it lands wherever noise is
    largest -- in the published runs, at 91-98% depth for four of nine models, and
    at a median AUROC of 1.000 for one of them. Cross-validation gives each concept
    all of its examples out-of-fold, and a shared layer replaces 25 independent
    selections with one, which also makes the resulting z-scores comparable across
    concepts.
    """
    scores = cross_validated_scores(acts, concepts, all_concepts, band,
                                    args.cv, args.cv_seed, args.method, args.variant)
    emo_concepts = [c for c in all_concepts if c in EMOTIONS]

    # per-layer mean AUROC across the emotion concepts -> the shared layer
    from oncoemotion.probing.probe import _auroc
    per_layer_mean, per_layer_auroc = [], []
    for li, l in enumerate(band):
        aus = {}
        for c in all_concepts:
            y = (concepts == c).astype(int)
            s = scores[c][li]
            ok = ~np.isnan(s)
            au = _auroc(s[ok], y[ok]) if ok.sum() and len(set(y[ok])) > 1 else float("nan")
            aus[c] = None if np.isnan(au) else float(au)
        per_layer_auroc.append(aus)
        vals = [aus[c] for c in emo_concepts if aus[c] is not None]
        per_layer_mean.append(float(np.mean(vals)) if vals else 0.0)
    best_li = int(np.argmax(per_layer_mean))
    shared = band[best_li]

    report.update({
        "eval": f"{args.cv}-fold cross-validated, one-vs-rest",
        "layer_policy": "single shared layer across all concepts",
        "shared_layer": shared,
        "shared_layer_depth": round(shared / (n_layers - 1), 3),
        "mean_emotion_auroc_by_layer": {str(l): round(m, 4) for l, m in zip(band, per_layer_mean)},
        "selection": (f"one layer selected over {len(band)} candidates by mean out-of-fold "
                      f"AUROC across {len(emo_concepts)} emotion concepts; per-concept "
                      f"AUROC below is out-of-fold at that layer"),
    })
    for c in all_concepts:
        y = (concepts == c).astype(int)
        s = scores[c][best_li]
        ok = ~np.isnan(s)
        res = evaluate_direction(np.asarray(s[ok]).reshape(-1, 1), y[ok], np.array([1.0]))
        key = vec_key(V, c, args.method, args.variant)
        report["concepts"][c] = {
            "kind": "emotion" if c in EMOTIONS else "control",
            "vector_key": key,
            "best_layer": shared,
            "best_layer_depth": round(shared / (n_layers - 1), 3),
            "best_auroc": res["auroc"],
            "best_auroc_ci": res["auroc_ci"],
            "best_cohens_d": res["cohens_d"],
            "n_pos_test": int(y[ok].sum()),
            "selection_layer_source": "shared, cross-validated",
            "auroc_by_layer": {str(l): per_layer_auroc[li][c] for li, l in enumerate(band)},
        }

    report["lexical_gate"] = _lexical_gate(V, args, shared)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report -> {args.report}")

    _print_lexical_gate(report["lexical_gate"])
    print(f"\nbanda layer {report['layer_band']} di {n_layers} | {args.cv}-fold CV")
    print(f"layer condiviso scelto: {shared} (profondita {report['shared_layer_depth']}), "
          f"AUROC media emozioni {per_layer_mean[best_li]:.3f}")
    print(f"\n{'concept':24} {'kind':8} {'AUROC':>7} {'CI':>16} {'d':>6} {'n_pos':>6}")
    order = sorted(report["concepts"], key=lambda c: -(report["concepts"][c]["best_auroc"] or 0))
    for c in order:
        r = report["concepts"][c]
        ci = r["best_auroc_ci"]
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci and ci[0] is not None else "-"
        print(f"{c:24} {r['kind']:8} {(r['best_auroc'] or 0):>7.3f} {ci_s:>16} "
              f"{r['best_cohens_d']:>6.2f} {r['n_pos_test']:>6}")
    below = [c for c in order if (report["concepts"][c]["best_auroc"] or 0) < 0.6]
    if below:
        print(f"\n[!] {len(below)}/{len(order)} concetti sotto AUROC 0.60 fuori campione: "
              f"{', '.join(below[:10])}{' ...' if len(below) > 10 else ''}")
        print("    queste direzioni non separano esempi mai visti: non vanno usate come assi.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", type=Path, default=_ROOT / "outputs/activations/emotion_acts.npz")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--figure", type=Path, default=_ROOT / "outputs/figures/layer_sweep_auroc.png")
    ap.add_argument("--band-lo", type=float, default=0.25,
                    help="lower edge of the layer search band, as a fraction of depth")
    ap.add_argument("--band-hi", type=float, default=0.75,
                    help="upper edge of the layer search band, as a fraction of depth")
    ap.add_argument("--min-n-pos", type=int, default=5,
                    help="warn when a concept has fewer positives than this in the test split")
    ap.add_argument("--cv", type=int, default=5,
                    help="k-fold cross-validated evaluation with one shared layer (0 = off, "
                         "use the single-split per-concept path instead)")
    ap.add_argument("--cv-seed", type=int, default=12345)
    args = ap.parse_args()

    A = np.load(args.acts, allow_pickle=True)
    acts, concepts, splits = A["acts"], A["concepts"], A["splits"]
    V = np.load(args.vecs, allow_pickle=True)
    n_layers = acts.shape[1]
    val = splits == "validation"
    test = splits == "test"

    # Search band. Layer 0 is the embedding output: at point E the same token ends
    # every prompt, so a direction picked there has zero projection variance and
    # zscore() returns a hard 0.0 for that concept in every row -- which then reads
    # as "the role does not move this emotion". Layers near the top pick up output
    # token identity rather than a concept. The band keeps the search where emotion
    # representations are reported to live (~50% depth, arXiv:2604.04064).
    lo = max(1, int(round(args.band_lo * (n_layers - 1))))
    hi = min(n_layers - 1, int(round(args.band_hi * (n_layers - 1))))
    band = list(range(lo, hi + 1))
    nested = bool(val.sum())

    report = {"model_id": str(A["model_id"]), "method": args.method, "variant": args.variant,
              "n_layers": n_layers, "eval": "one_vs_rest_heldout",
              "layer_band": [lo, hi], "layer_band_fraction": [args.band_lo, args.band_hi],
              "selection": ("layer chosen on the validation split, metrics reported on "
                            "the test split") if nested else
                           ("no validation split available: layer chosen and reported on "
                            "the test split (metrics are selection-optimistic)"),
              "concepts": {}}
    sweep, best_vec = {}, {}
    all_concepts = sorted(set(concepts.tolist()) - {"neutral"})

    if args.cv:
        return _run_cv(args, acts, concepts, all_concepts, band, n_layers, V, report)

    for concept in all_concepts:
        key = vec_key(V, concept, args.method, args.variant)
        if key not in V:
            continue
        layer_vecs = V[key]
        idx_test = np.where(test)[0]
        y_test = (concepts[idx_test] == concept).astype(int)
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        # full sweep on the test split: descriptive, drives the figure. No bootstrap
        # here -- only the selected layer gets an interval.
        per_layer, aurocs = [], []
        for l in range(n_layers):
            res = evaluate_direction(acts[idx_test, l, :], y_test, layer_vecs[l], n_boot=0)
            per_layer.append({"layer": l, **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                             for k, v in res.items()}})
            aurocs.append(res["auroc"] if not np.isnan(res["auroc"]) else 0.0)

        # selection: on the validation split when there is one, so the reported
        # AUROC is not the maximum of the same numbers it is reported from
        if nested:
            idx_val = np.where(val)[0]
            y_val = (concepts[idx_val] == concept).astype(int)
            if y_val.sum() == 0 or y_val.sum() == len(y_val):
                sel_scores = [aurocs[l] for l in band]
            else:
                sel_scores = [evaluate_direction(acts[idx_val, l, :], y_val, layer_vecs[l],
                                                 n_boot=0)["auroc"] for l in band]
                sel_scores = [0.0 if s is None or np.isnan(s) else s for s in sel_scores]
        else:
            sel_scores = [aurocs[l] for l in band]
        best_l = band[int(np.argmax(sel_scores))]

        # the selected layer is the only one that gets a bootstrap interval
        best_res = evaluate_direction(acts[idx_test, best_l, :], y_test, layer_vecs[best_l])
        per_layer[best_l] = {"layer": best_l,
                             **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                for k, v in best_res.items()}}

        # what the old unconstrained argmax-on-test would have reported, kept so the
        # size of the selection optimism is visible instead of assumed
        naive_l = int(np.argmax([p["auroc"] if p["auroc"] is not None else -1 for p in per_layer]))

        sweep[concept] = aurocs
        best_vec[concept] = layer_vecs[best_l]
        report["concepts"][concept] = {
            "kind": "emotion" if concept in EMOTIONS else "control",
            "vector_key": key,
            "best_layer": best_l,
            "best_layer_depth": round(best_l / (n_layers - 1), 3),
            "best_auroc": per_layer[best_l]["auroc"],
            "best_auroc_ci": per_layer[best_l]["auroc_ci"],
            "best_cohens_d": per_layer[best_l]["cohens_d"],
            "n_pos_test": int(y_test.sum()),
            "selection_layer_source": "validation" if nested else "test",
            "unconstrained_argmax_layer": naive_l,
            "unconstrained_argmax_auroc": per_layer[naive_l]["auroc"],
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

    print(f"\nlayer band {report['layer_band']} of {n_layers} | {report['selection']}")
    print(f"\n{'concept':24} {'kind':8} {'best_L':>6} {'AUROC':>7} {'CI':>16} {'d':>6} "
          f"{'n_pos':>6} {'argmax_L':>9} {'argmax_AUROC':>13}")
    for c in sorted(report["concepts"]):
        r = report["concepts"][c]
        ci = r["best_auroc_ci"]
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci and ci[0] is not None else "-"
        au = r["best_auroc"] or 0
        print(f"{c:24} {r['kind']:8} {r['best_layer']:>6} {au:>7.3f} {ci_s:>16} "
              f"{r['best_cohens_d']:>6.2f} {r['n_pos_test']:>6} {r['unconstrained_argmax_layer']:>9} "
              f"{(r['unconstrained_argmax_auroc'] or 0):>13.3f}")

    thin = [c for c, r in report["concepts"].items() if r["n_pos_test"] < args.min_n_pos]
    if thin:
        print(f"\n[!] {len(thin)} concetti con meno di {args.min_n_pos} positivi nel test split: "
              f"{', '.join(sorted(thin)[:8])}{' ...' if len(thin) > 8 else ''}")
        print("    la AUROC su cosi pochi positivi ha un intervallo molto ampio; "
              "allargare i seed in seeds.py.")
    naive_gap = [abs((r['best_auroc'] or 0) - (r['unconstrained_argmax_auroc'] or 0))
                 for r in report["concepts"].values()]
    if naive_gap:
        print(f"\nottimismo da selezione evitato: la AUROC riportata e in media "
              f"{np.mean(naive_gap):.3f} sotto l'argmax non vincolato sullo stesso split.")

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
