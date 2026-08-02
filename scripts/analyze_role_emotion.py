#!/usr/bin/env python
"""Aggregate the role x emotion run into the study's answers.

Reads ``outputs/role_emotion/<slug>__rows.jsonl`` and computes:
  A. emotionality by role  (mean z of the negative-affect axis at point E);
  B. labelling accuracy on EXACT (term) items, per role x framing x ablation,
     with the deterministic mapper as a reference baseline;
  C. false-positive coding on abstain items (model forced to pick a term):
     confidence + FP-rate per cell;
  D. emotion-vs-error link (emotion z on correct vs wrong; point-biserial r);
  E. framing effect (emotional vs neutral, paired);
  F. ablation effect (label flips intact->ablated; accuracy delta);
  G. PRIMARY ENDPOINT — role x framing interaction.

E and G are not the same question. E pools the roles and therefore measures a
property of the patient's text: emotional phrasing is harder to code. G asks
whether the *assigned role* changes how much that phrasing costs, which is the
claim the study actually makes. Everything in G is paired within item and within
role. The per-model contrast is descriptive; the single inferential test lives in
``analyze_results.py``, which pools every model.

Writes ``<slug>__analysis.json`` and prints a summary.

Usage:
    python scripts/analyze_role_emotion.py --rows outputs/role_emotion/qwen2.5-3b-instruct__rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

NEG_AFFECT = ["afraid_alarmed", "anxious_nervous", "sad"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(mean(xs)) if xs else None


def _emo_composite(z: dict) -> float:
    vals = [z[c] for c in NEG_AFFECT if c in z]
    return float(mean(vals)) if vals else 0.0


def _point_biserial(binary, cont):
    """r between a 0/1 label and a continuous score."""
    b = np.asarray(binary, float); c = np.asarray(cont, float)
    if len(b) < 3 or c.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(b, c)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fp-threshold", type=float, default=0.5,
                    help="softmax_top1 above which a forced code counts as a false-positive")
    ap.add_argument("--reference-role", default="none",
                    help="baseline role the primary contrast is taken against "
                         "(the no-role control, so a positive contrast means the role protects)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.rows.read_text(encoding="utf-8").splitlines() if l.strip()]
    roles = sorted({r["role"] for r in rows})
    for r in rows:
        r["emo"] = _emo_composite(r["z"])

    term = [r for r in rows if r["gold_class"] == "term"]
    abst = [r for r in rows if r["gold_class"] == "abstain"]

    # ---- A. emotionality by role (intact only; split by framing) ----
    A = {}
    for role in roles:
        cell = {}
        for fr in ("neutral", "emotional", "all"):
            sub = [r for r in rows if r["role"] == role and not r["ablated"]
                   and (fr == "all" or r["framing"] == fr)]
            cell[fr] = {
                "emo_z": _mean([r["emo"] for r in sub]),
                **{c: _mean([r["z"][c] for r in sub]) for c in NEG_AFFECT},
                "n": len(sub),
            }
        A[role] = cell

    # ---- B. accuracy on term items, per role x framing x ablation ----
    def acc(sub):
        c = [r["correct"] for r in sub if r["correct"] is not None]
        return (round(sum(c) / len(c), 4), len(c)) if c else (None, 0)

    B = {}
    for role in roles:
        B[role] = {}
        for abl in (False, True):
            for fr in ("neutral", "emotional", "all"):
                sub = [r for r in term if r["role"] == role and r["ablated"] == abl
                       and (fr == "all" or r["framing"] == fr)]
                a, n = acc(sub)
                B[role][f"{'ablated' if abl else 'intact'}|{fr}"] = {"acc": a, "n": n}
    # mapper reference accuracy (condition-independent; use intact/oncologo rows once per record)
    seen, mp_correct = set(), []
    for r in term:
        if r["record_id"] in seen:
            continue
        seen.add(r["record_id"])
        mp_correct.append(int(r.get("mapper_pro_id") == r["gold_pro_id"]))
    mapper_term_acc = round(sum(mp_correct) / len(mp_correct), 4) if mp_correct else None

    # ---- C. false-positive coding on abstain items ----
    C = {}
    for role in roles:
        C[role] = {}
        for abl in (False, True):
            for fr in ("neutral", "emotional", "all"):
                sub = [r for r in abst if r["role"] == role and r["ablated"] == abl
                       and (fr == "all" or r["framing"] == fr)]
                conf = [r.get("model_map_score", 0.0) for r in sub]
                # false-positive coding = the model mapped a PRO term to an item
                # that should abstain
                fp = [1 if r.get("model_matched") else 0 for r in sub]
                C[role][f"{'ablated' if abl else 'intact'}|{fr}"] = {
                    "mean_conf": _mean(conf),
                    "fp_rate": round(sum(fp) / len(fp), 4) if fp else None,
                    "n": len(sub),
                }

    # ---- D. emotion vs error (term items, intact) ----
    intact_term = [r for r in term if not r["ablated"]]
    wrong = [r for r in intact_term if r["correct"] is False]
    right = [r for r in intact_term if r["correct"] is True]
    D = {
        "emo_z_on_wrong": _mean([r["emo"] for r in wrong]),
        "emo_z_on_correct": _mean([r["emo"] for r in right]),
        "point_biserial_error_vs_emo": _point_biserial(
            [0 if r["correct"] else 1 for r in intact_term],
            [r["emo"] for r in intact_term]),
        "n_wrong": len(wrong), "n_correct": len(right),
    }

    # ---- E. framing effect (paired by pair_id, intact, term) ----
    by_pair = {}
    for r in intact_term:
        by_pair.setdefault(r["pair_id"], {})[r["framing"]] = r
    flips_fr, both = 0, 0
    neu_acc, emo_acc = [], []
    for pid, d in by_pair.items():
        if "neutral" in d and "emotional" in d:
            both += 1
            neu_acc.append(int(bool(d["neutral"]["correct"])))
            emo_acc.append(int(bool(d["emotional"]["correct"])))
            if d["neutral"]["model_top1_id"] != d["emotional"]["model_top1_id"]:
                flips_fr += 1
    E = {
        "neutral_acc": round(_mean(neu_acc), 4) if neu_acc else None,
        "emotional_acc": round(_mean(emo_acc), 4) if emo_acc else None,
        "label_flips_neutral_vs_emotional": flips_fr,
        "n_pairs": both,
    }

    # ---- G. PRIMARY ENDPOINT: role x framing interaction ----
    # E above pools the roles, so it answers "does emotional phrasing hurt?" -- a
    # property of the patient's text. The study's claim is about the role, so the
    # quantifier has to be the role: does the assigned role modulate how much the
    # emotional phrasing hurts? Everything here is within item and within role, so
    # the item is its own control on both factors.
    ref = args.reference_role if args.reference_role in roles else (
        "none" if "none" in roles else roles[0])
    per_role = {}
    for role in roles:
        pairs = {}
        for r in term:
            if r["role"] == role and not r["ablated"]:
                pairs.setdefault(r["pair_id"], {})[r["framing"]] = r
        neu, emo, b, c = [], [], 0, 0
        for pid, d in pairs.items():
            if "neutral" not in d or "emotional" not in d:
                continue
            n_ok, e_ok = bool(d["neutral"]["correct"]), bool(d["emotional"]["correct"])
            neu.append(int(n_ok)); emo.append(int(e_ok))
            b += int(n_ok and not e_ok)
            c += int(e_ok and not n_ok)
        per_role[role] = {
            "neutral_acc": round(_mean(neu), 4) if neu else None,
            "emotional_acc": round(_mean(emo), 4) if emo else None,
            "delta": round(_mean(emo) - _mean(neu), 4) if neu else None,
            "n_pairs": len(neu),
            "discordant_neutral_only": b,
            "discordant_emotional_only": c,
            # per-item framing effect, kept so the pooled model and the
            # within-model sign test read the same numbers
            "item_deltas": {pid: int(bool(d["emotional"]["correct"])) - int(bool(d["neutral"]["correct"]))
                            for pid, d in pairs.items() if len(d) == 2},
        }

    contrasts = {}
    ref_deltas = per_role.get(ref, {}).get("item_deltas", {})
    for role in roles:
        if role == ref:
            continue
        rd = per_role[role]["item_deltas"]
        shared = sorted(set(rd) & set(ref_deltas))
        diffs = [rd[p] - ref_deltas[p] for p in shared]
        pos = sum(1 for x in diffs if x > 0)
        neg = sum(1 for x in diffs if x < 0)
        contrasts[role] = {
            # >0 means the role protects: the framing penalty is smaller than the
            # no-role control's
            "delta_difference": round(_mean(diffs), 4) if diffs else None,
            "n_items": len(diffs),
            "items_role_better": pos,
            "items_role_worse": neg,
            "items_tied": len(diffs) - pos - neg,
        }

    # resolution floor: with n paired items one discordant item moves the delta by
    # 1/n, so an effect below that is not measurable however it is tested
    n_pairs_ref = per_role.get(ref, {}).get("n_pairs") or 0
    G = {
        "primary_endpoint": "role x framing interaction on term-item accuracy",
        "reference_role": ref,
        "by_role": {k: {kk: vv for kk, vv in v.items() if kk != "item_deltas"}
                    for k, v in per_role.items()},
        "contrasts_vs_reference": contrasts,
        "resolution_floor": round(1.0 / n_pairs_ref, 4) if n_pairs_ref else None,
    }

    # ---- F. ablation effect (same record intact vs ablated) ----
    idx = {}
    for r in rows:
        idx.setdefault((r["record_id"], r["role"]), {})[r["ablated"]] = r
    flips_abl, comp = 0, 0
    acc_intact, acc_ablate = [], []
    for k, d in idx.items():
        if False in d and True in d:
            comp += 1
            if d[False]["model_top1_id"] != d[True]["model_top1_id"]:
                flips_abl += 1
            if d[False]["gold_class"] == "term":
                acc_intact.append(int(bool(d[False]["correct"])))
                acc_ablate.append(int(bool(d[True]["correct"])))
    F = {
        "label_flips_intact_vs_ablated": flips_abl,
        "flip_rate": round(flips_abl / comp, 4) if comp else None,
        "term_acc_intact": round(_mean(acc_intact), 4) if acc_intact else None,
        "term_acc_ablated": round(_mean(acc_ablate), 4) if acc_ablate else None,
        "n_compared": comp,
    }

    analysis = {
        "rows_file": str(args.rows), "n_rows": len(rows), "roles": roles,
        "n_term": len(term), "n_abstain": len(abst),
        "A_emotionality_by_role": A,
        "B_accuracy_term": B, "mapper_term_accuracy": mapper_term_acc,
        "C_false_positive_abstain": C,
        "D_emotion_vs_error": D,
        "E_framing_effect": E,
        "F_ablation_effect": F,
        "G_role_by_framing": G,
        "fp_threshold": args.fp_threshold,
    }
    out = args.out or args.rows.with_name(args.rows.name.replace("__rows.jsonl", "__analysis.json"))
    out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- console summary ----
    print(f"\n=== Role x Emotion — {args.rows.name} ===")
    print(f"rows={len(rows)}  term={len(term)}  abstain={len(abst)}  roles={roles}")
    print("\nA) Emotionality (neg-affect z at point E, intact):")
    for role in roles:
        a = A[role]
        print(f"   {role:10} all={a['all']['emo_z']:+.3f}  "
              f"neutral={a['neutral']['emo_z']:+.3f}  emotional={a['emotional']['emo_z']:+.3f}")
    print(f"\nB) Model term-accuracy (intact|all) vs mapper {mapper_term_acc}:")
    for role in roles:
        print(f"   {role:10} intact={B[role]['intact|all']['acc']}  ablated={B[role]['ablated|all']['acc']}"
              f"  (emo-framing intact={B[role]['intact|emotional']['acc']} vs neutral={B[role]['intact|neutral']['acc']})")
    print("\nC) False-positive coding on abstain (mean_conf / fp_rate, intact|all):")
    for role in roles:
        c = C[role]['intact|all']
        print(f"   {role:10} conf={c['mean_conf']:.3f}  fp_rate={c['fp_rate']}")
    print(f"\nD) Emotion vs error: emo_z wrong={D['emo_z_on_wrong']} vs correct={D['emo_z_on_correct']} "
          f"| point-biserial(err,emo)={D['point_biserial_error_vs_emo']}")
    print(f"E) Framing: acc neutral={E['neutral_acc']} vs emotional={E['emotional_acc']} "
          f"| label flips={E['label_flips_neutral_vs_emotional']}/{E['n_pairs']}")
    print(f"F) Ablation: term acc intact={F['term_acc_intact']} vs ablated={F['term_acc_ablated']} "
          f"| label flips={F['label_flips_intact_vs_ablated']}/{F['n_compared']} ({F['flip_rate']})")
    print(f"\nG) PRIMARY - role x framing (reference role: {G['reference_role']}, "
          f"resolution floor {G['resolution_floor']}):")
    for role in roles:
        g = G["by_role"][role]
        mark = "  <- ref" if role == G["reference_role"] else ""
        print(f"   {role:10} {g['neutral_acc']} -> {g['emotional_acc']}  "
              f"delta={g['delta']:+.4f}  (n={g['n_pairs']}, disc {g['discordant_neutral_only']}/"
              f"{g['discordant_emotional_only']}){mark}")
    for role, ct in G["contrasts_vs_reference"].items():
        flag = "" if ct["delta_difference"] is None or G["resolution_floor"] is None else (
            "" if abs(ct["delta_difference"]) >= G["resolution_floor"] else "  [sotto la risoluzione]")
        print(f"   contrasto {role} vs {G['reference_role']}: {ct['delta_difference']:+.4f} "
              f"(meglio {ct['items_role_better']} / peggio {ct['items_role_worse']} / "
              f"pari {ct['items_tied']}){flag}")
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
