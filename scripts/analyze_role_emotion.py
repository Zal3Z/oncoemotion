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
  F. ablation effect (label flips intact->ablated; accuracy delta).

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
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
