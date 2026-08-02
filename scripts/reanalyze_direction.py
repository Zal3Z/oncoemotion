#!/usr/bin/env python
"""Re-analyse the C2 direction result by rank instead of by threshold.

Why this exists
---------------
C2 was reported as "cosine with the fear axis below 0.2 in 9/9 models", read as
evidence that the role-induced shift is *not* fear. That reading does not survive
contact with the numbers: across 9 models x 25 emotion axes the largest absolute
cosine observed is 0.186, so **every** axis is below 0.2 in **every** model. The
threshold cannot fail, which means it tested nothing.

An absolute cosine in a several-thousand-dimensional space has no natural scale.
Two quantities do:

  * the **rank** of the fear axis among all emotion axes for the same shift, and
  * its magnitude relative to the **median** axis.

On both, fear is not low: it ranks 4th of 25 in six of nine models, with |cos|
typically 2-4x the median. The corrected statement is therefore the opposite of
the original one -- the shift is more aligned with the fear/anxiety cluster than
with 20 of the 25 axes, in a regime where every absolute alignment is small.

This script runs on the spectrum JSON already on disk, so it needs no GPU. The
random-direction null (``null_cos`` / ``z_vs_random``) needs the persona hidden
states, which are not persisted; ``run_role_spectrum.py`` now computes it during
the next run and this script reports it when present.

Usage:
    python scripts/reanalyze_direction.py --dir outputs/role_spectrum
    python scripts/reanalyze_direction.py --dir "oncoemotion_results5/outputs/role_spectrum"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

CONTRASTS = ("profani_minus_medici", "profani_minus_tecnici", "tecnici_minus_medici")
FEAR = "afraid_alarmed"
# The historical claim under test.
LEGACY_THRESHOLD = 0.2


def _summarize(spec: dict, contrast: str) -> dict | None:
    key = f"{contrast}_cos"
    emo = set(spec.get("emo_concepts") or [])
    vals = {c: v[key] for c, v in spec.get("direction", {}).items()
            if c in emo and v.get(key) is not None}
    if not vals or FEAR not in vals:
        return None
    order = sorted(vals, key=lambda c: -abs(vals[c]))
    med = median(abs(v) for v in vals.values())
    fear = vals[FEAR]
    return {
        "n_axes": len(vals),
        "fear_cos": fear,
        "fear_rank": order.index(FEAR) + 1,
        "fear_abs_over_median": round(abs(fear) / med, 2) if med > 1e-12 else None,
        "median_abs_cos": round(med, 4),
        "max_abs_cos": round(max(abs(v) for v in vals.values()), 4),
        "top_axis": order[0],
        "top_axis_cos": vals[order[0]],
        "ranked_axes": order,
        # present only after a run with the updated run_role_spectrum.py
        "z_vs_random": (spec.get("direction_summary", {}).get(contrast, {})
                        .get("per_axis", {}).get(FEAR, {}).get("z_vs_random")),
        "anisotropy_at_fear_layer": spec.get("direction", {}).get(FEAR, {}).get("anisotropy_at_layer"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True,
                    help="directory holding <slug>__spectrum.json files")
    ap.add_argument("--contrast", default="profani_minus_medici", choices=CONTRASTS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.dir.glob("*__spectrum.json"))
    if not files:
        print(f"no spectrum files under {args.dir}")
        return 1

    per_model, global_max = {}, 0.0
    for f in files:
        spec = json.loads(f.read_text(encoding="utf-8"))
        s = _summarize(spec, args.contrast)
        if s is None:
            continue
        slug = f.name.split("__")[0]
        per_model[slug] = s
        global_max = max(global_max, s["max_abs_cos"])

    print(f"\n=== C2 rivisto - contrasto {args.contrast} ===")
    print(f"{'modello':28s} {'cos paura':>10s} {'rango':>8s} {'|cos|/med':>10s} "
          f"{'mediana':>9s} {'asse in cima':>22s}")
    for slug, s in per_model.items():
        print(f"{slug:28s} {s['fear_cos']:10.3f} {s['fear_rank']:4d}/{s['n_axes']:<3d} "
              f"{s['fear_abs_over_median']:10.2f} {s['median_abs_cos']:9.4f} "
              f"{s['top_axis']:>16s} {s['top_axis_cos']:5.3f}")

    n = len(per_model)
    top_quartile = sum(1 for s in per_model.values() if s["fear_rank"] <= max(1, s["n_axes"] // 4))
    above_median = sum(1 for s in per_model.values()
                       if abs(s["fear_cos"]) > s["median_abs_cos"])
    print(f"\n  asse paura nel quartile piu allineato: {top_quartile}/{n} modelli")
    print(f"  asse paura sopra la mediana degli assi: {above_median}/{n} modelli")
    print(f"  massimo |cos| su tutti i modelli e tutti gli assi: {global_max:.4f}")
    if global_max < LEGACY_THRESHOLD:
        print(f"  => la soglia storica di {LEGACY_THRESHOLD} e superata da ZERO assi su "
              f"{sum(s['n_axes'] for s in per_model.values())} combinazioni modello x asse.")
        print("     'sotto 0.2 in 9/9' e vero per ogni asse: non e un test, e la scala.")

    zs = [s["z_vs_random"] for s in per_model.values() if s["z_vs_random"] is not None]
    if zs:
        print(f"\n  z contro direzioni casuali disponibile per {len(zs)}/{n} modelli, "
              f"mediana {median(zs):+.1f}")
    else:
        print("\n  z contro direzioni casuali non disponibile: gli stati nascosti non sono")
        print("  salvati nei run esistenti. Arriva col prossimo run di run_role_spectrum.py.")

    ani = [s["anisotropy_at_fear_layer"] for s in per_model.values()
           if s["anisotropy_at_fear_layer"] is not None]
    if ani:
        flagged = [k for k, s in per_model.items()
                   if (s["anisotropy_at_fear_layer"] or 0) > 0.95]
        print(f"  anisotropia al layer di estrazione: mediana {median(ani):.3f}"
              + (f", oltre 0.95 in {flagged}" if flagged else ", nessun modello oltre 0.95"))

    out = args.out or (args.dir / f"direction_reanalysis__{args.contrast}.json")
    out.write_text(json.dumps({
        "contrast": args.contrast,
        "legacy_threshold": LEGACY_THRESHOLD,
        "global_max_abs_cos": round(global_max, 4),
        "legacy_threshold_is_vacuous": global_max < LEGACY_THRESHOLD,
        "n_models": n,
        "fear_in_top_quartile": top_quartile,
        "fear_above_median": above_median,
        "per_model": per_model,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
