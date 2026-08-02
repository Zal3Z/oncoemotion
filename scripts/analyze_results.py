#!/usr/bin/env python
"""Pooled inferential analysis across models. ONE primary test.

Until now the repository had no inferential statistics at all: no confidence
interval on any accuracy, no test on the framing, role or ablation effect, a
``benjamini_hochberg`` helper that nothing ever called, and a single deterministic
run per model so not even a run-to-run variance estimate existed. This script is
that layer.

Primary endpoint
----------------
The role x framing interaction on term-item coding accuracy, pooled over models.

Model: **conditional logistic regression**, stratified by (item x model). The design
is paired twice over -- the same clinical content appears in two framings, and the
same item is seen under every role -- so conditioning on the stratum removes the
item and the model as nuisance parameters exactly, rather than estimating a
variance component for them. That matters here: nine models are far too few to
estimate a level-2 variance, which is why the model enters as part of the stratum
and not as a random intercept.

Conditional logit uses only strata that are not constant in the outcome, so it
reduces to the discordant observations -- the same ones the per-model contrast in
``analyze_role_emotion.py`` counts. The two views therefore report the same
evidence, one descriptively and one with an interval.

Not done, deliberately
----------------------
No per-model McNemar with a Benjamini-Hochberg pass. With 35-112 pairs per cell and
discordant counts in the single digits, most per-model tests are underpowered; a
table of nulls would be read against a hypothesis that is not the study's. The
claim is consistency across models, and that is what the pooled model tests. Per
model this script reports estimates and intervals, no p-values.

Benjamini-Hochberg is applied only to the pre-declared secondary family.

Usage:
    python scripts/analyze_results.py --rows-glob "outputs/role_emotion/*__rows.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.statistics import (  # noqa: E402
    benjamini_hochberg, bh_adjusted_pvalues, bootstrap_ci)

REFERENCE_ROLE = "none"
SECONDARY_FAMILY = ("framing_main_effect", "ablation_emotion_vs_random",
                    "framing_effect_on_mapper")


def _load(paths) -> list[dict]:
    rows = []
    for p in paths:
        slug = Path(p).name.split("__")[0]
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["model"] = slug
            r.setdefault("arm", "emotion" if r.get("ablated") else "intact")
            rows.append(r)
    return rows


def _design(rows, roles):
    """Long-format design for the interaction, one row per (item, role, framing)."""
    keep = [r for r in rows if r["gold_class"] == "term" and r["arm"] == "intact"
            and r.get("correct") is not None and r["role"] in roles]
    y, emo, strata = [], [], []
    role_d = {ro: [] for ro in roles if ro != REFERENCE_ROLE}
    for r in keep:
        y.append(int(bool(r["correct"])))
        emo.append(1 if r["framing"] == "emotional" else 0)
        for ro in role_d:
            role_d[ro].append(1 if r["role"] == ro else 0)
        # the stratum is the item within the model: both are conditioned out
        strata.append(f"{r['model']}|{r['pair_id']}")
    return (np.array(y), np.array(emo),
            {k: np.array(v) for k, v in role_d.items()}, np.array(strata))


def _fit_primary(rows, roles) -> dict:
    from statsmodels.discrete.conditional_models import ConditionalLogit

    y, emo, role_d, strata = _design(rows, roles)
    names, cols = ["emotional"], [emo]
    for ro, d in role_d.items():
        names.append(f"role[{ro}]")
        cols.append(d)
    for ro, d in role_d.items():
        names.append(f"emotional x role[{ro}]")
        cols.append(emo * d)
    X = np.column_stack(cols).astype(float)

    res = ConditionalLogit(y, X, groups=strata).fit(disp=0)
    ci = res.conf_int()
    informative = sum(1 for s in set(strata.tolist())
                      if 0 < y[strata == s].sum() < int((strata == s).sum()))
    out = {
        "model": "conditional logistic, strata = item x model",
        "reference_role": REFERENCE_ROLE,
        "n_observations": int(len(y)),
        "n_strata": int(len(set(strata.tolist()))),
        "n_informative_strata": int(informative),
        "terms": {},
    }
    for i, nm in enumerate(names):
        b = float(res.params[i])
        lo, hi = float(ci[i][0]), float(ci[i][1])
        out["terms"][nm] = {
            "log_odds": round(b, 4),
            "odds_ratio": round(float(np.exp(b)), 4),
            "ci95_odds_ratio": [round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)],
            "se": round(float(res.bse[i]), 4),
            "p": round(float(res.pvalues[i]), 5),
        }
    inter = [n for n in names if n.startswith("emotional x")]
    out["primary_terms"] = inter
    out["primary_p"] = min((out["terms"][n]["p"] for n in inter), default=None)
    return out


def _per_model_contrasts(rows, roles) -> dict:
    """Descriptive per-model interaction, with a bootstrap interval. No p-values."""
    per = {}
    for m in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == m and r["gold_class"] == "term"
               and r["arm"] == "intact" and r.get("correct") is not None]
        deltas = {}
        for ro in roles:
            pairs = {}
            for r in sub:
                if r["role"] == ro:
                    pairs.setdefault(r["pair_id"], {})[r["framing"]] = bool(r["correct"])
            deltas[ro] = {p: int(d["emotional"]) - int(d["neutral"])
                          for p, d in pairs.items() if len(d) == 2}
        ref = deltas.get(REFERENCE_ROLE, {})
        entry = {}
        for ro in roles:
            if ro == REFERENCE_ROLE or not ref:
                continue
            shared = sorted(set(deltas.get(ro, {})) & set(ref))
            if len(shared) < 3:
                continue
            diff = np.array([deltas[ro][p] - ref[p] for p in shared], float)
            _, lo, hi = bootstrap_ci(diff, statistic=np.mean, n_boot=2000, seed=12345)
            entry[f"{ro} vs {REFERENCE_ROLE}"] = {
                "mean_difference": round(float(np.mean(diff)), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "n_items": len(diff),
                "resolution_floor": round(1.0 / len(diff), 4),
            }
        per[m] = entry
    return per


def _secondaries(rows, roles) -> dict:
    """Pre-declared secondary family, corrected together with Benjamini-Hochberg."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    out, pvals, keys = {}, [], []

    # 1. framing main effect, pooled -- a property of the text, not of the role
    y, emo, _, strata = _design(rows, roles)
    r1 = ConditionalLogit(y, emo.reshape(-1, 1).astype(float), groups=strata).fit(disp=0)
    out["framing_main_effect"] = {"odds_ratio": round(float(np.exp(r1.params[0])), 4),
                                  "p": float(r1.pvalues[0])}
    pvals.append(float(r1.pvalues[0])); keys.append("framing_main_effect")

    # 2. the mechanistic gate: does ablating the emotion directions flip labels more
    #    than ablating norm- and layer-matched random ones?
    idx = {}
    for r in rows:
        idx.setdefault((r["model"], r["record_id"], r["role"]), {})[r["arm"]] = r
    fe, fr = [], []
    for d in idx.values():
        if "intact" not in d:
            continue
        if "emotion" in d:
            fe.append(int(d["intact"]["model_top1_id"] != d["emotion"]["model_top1_id"]))
        if "random" in d:
            fr.append(int(d["intact"]["model_top1_id"] != d["random"]["model_top1_id"]))
    if fe and fr:
        fe_a, fr_a = np.asarray(fe, float), np.asarray(fr, float)
        rng = np.random.default_rng(12345)
        boot = np.array([rng.choice(fe_a, len(fe_a)).mean() - rng.choice(fr_a, len(fr_a)).mean()
                         for _ in range(2000)])
        diff = float(fe_a.mean() - fr_a.mean())
        p = float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        out["ablation_emotion_vs_random"] = {
            "flip_rate_emotion": round(float(fe_a.mean()), 4),
            "flip_rate_random": round(float(fr_a.mean()), 4),
            "difference": round(diff, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "p": p,
            "gate_passes": bool(diff > 0 and lo > 0),
        }
        pvals.append(p); keys.append("ablation_emotion_vs_random")
    else:
        out["ablation_emotion_vs_random"] = {
            "gate_passes": None,
            "note": "random control arm absent: rerun with --arms intact emotion random"}

    # 3. the deterministic mapper must be framing-invariant on the template item set.
    #    If it is not, part of any framing effect in a model is surface difficulty
    #    rather than affective framing.
    seen, mn, me = set(), [], []
    for r in rows:
        if r["gold_class"] != "term" or r["record_id"] in seen:
            continue
        seen.add(r["record_id"])
        ok = int(r.get("mapper_pro_id") == r["gold_pro_id"])
        (me if r["framing"] == "emotional" else mn).append(ok)
    if mn and me:
        out["framing_effect_on_mapper"] = {
            "neutral": round(float(np.mean(mn)), 4),
            "emotional": round(float(np.mean(me)), 4),
            "difference": round(float(np.mean(me) - np.mean(mn)), 4),
            "note": ("a purely lexical system should be framing-invariant on the "
                     "template item set; a non-zero difference means the manipulation "
                     "leaks into surface form"),
        }

    if pvals:
        rej = benjamini_hochberg(pvals)
        adj = bh_adjusted_pvalues(pvals)
        for k, a, rj in zip(keys, adj, rej):
            out[k]["p_adjusted"] = round(float(a), 5)
            out[k]["significant_after_bh"] = bool(rj)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-glob", default=str(_ROOT / "outputs/role_emotion/*__rows.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=_ROOT / "outputs/reports/primary_analysis.json")
    ap.add_argument("--roles", nargs="+", default=["oncologo", "generico", "none"])
    args = ap.parse_args()

    paths = sorted(glob.glob(args.rows_glob))
    if not paths:
        print(f"no rows files matched {args.rows_glob}")
        return 1
    rows = _load(paths)
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} rows | {len(models)} models | roles={args.roles}")

    primary = _fit_primary(rows, args.roles)
    per_model = _per_model_contrasts(rows, args.roles)
    secondary = _secondaries(rows, args.roles)

    report = {"models": models, "n_rows": len(rows),
              "primary": primary, "per_model": per_model,
              "secondary": secondary, "secondary_family": list(SECONDARY_FAMILY)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== PRIMARIO: interazione ruolo x framing ===")
    print(f"{primary['model']} | {primary['n_observations']} osservazioni, "
          f"{primary['n_informative_strata']}/{primary['n_strata']} strati informativi")
    for nm, t in primary["terms"].items():
        star = "  <- primario" if nm in primary["primary_terms"] else ""
        print(f"  {nm:26s} OR={t['odds_ratio']:6.3f} "
              f"[{t['ci95_odds_ratio'][0]:.3f}, {t['ci95_odds_ratio'][1]:.3f}]  "
              f"p={t['p']:.4f}{star}")

    print("\n=== per modello (stime e intervalli, nessun p-value) ===")
    for m, e in per_model.items():
        for k, v in e.items():
            flag = "" if abs(v["mean_difference"]) >= v["resolution_floor"] \
                else "  [sotto la risoluzione]"
            print(f"  {m:26s} {k:22s} {v['mean_difference']:+.4f} "
                  f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] n={v['n_items']}{flag}")

    print("\n=== secondari (Benjamini-Hochberg sulla famiglia dichiarata) ===")
    shown = ("odds_ratio", "difference", "flip_rate_emotion", "flip_rate_random",
             "neutral", "emotional", "gate_passes")
    for k, v in secondary.items():
        if "p" not in v:
            print(f"  {k:28s} {v.get('note', '')}")
            continue
        p = v.get("p_adjusted", v.get("p"))
        body = "  ".join(f"{kk}={vv}" for kk, vv in v.items() if kk in shown)
        print(f"  {k:28s} {body}" + (f"  p_adj={p:.4f}" if isinstance(p, float) else ""))
    print(f"\nWrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
