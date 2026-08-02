#!/usr/bin/env python
"""[Role spectrum] WHY does the role change emotionality — and how systematically?

Places many personas on an emotionality scale at the PRO-CTCAE decision point and
decomposes the effect to tell apart competing explanations:

  * SPECTRUM   — mean emotion z (afraid/anxious/sad/calm) per persona. Is a doctor
                 low because of medical familiarity, or professional detachment?
                 The engineer/lawyer/accountant (technical, non-medical) are the
                 discriminating test.
  * SPECIFICITY— does the role move ONLY the affect axes, or everything (clinical
                 severity, negative valence)? Compared against control directions.
  * PERSONA vs REACTION — is the persona already (un)emotional on a NEUTRAL text
                 (before any symptom), or does the role change how it REACTS to the
                 symptom? baseline vs clinical, and reaction = clinical - baseline.
  * DIRECTION  — is the 'layperson - expert' shift ALIGNED with the fear axis?
                 cosine between the state-difference vector and the emotion
                 direction. If high, the role literally moves the state along fear.

All personas are measured at the identical teacher-forced point E; only the SYSTEM
persona varies. z-scores use a single fixed reference (neutral texts, no role), so
personas are comparable on one scale.

Usage:
    python scripts/run_role_spectrum.py --limit 8 --device cuda --dtype float16   # local pilot
    python scripts/run_role_spectrum.py --model <hf_id> --dtype bfloat16 --device auto \
        --vecs outputs/models/<slug>/emotion_vectors.npz \
        --val-report outputs/models/<slug>/vector_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.clinical.prompt import (  # noqa: E402
    build_decision_messages, build_padded_personas, TEACHER_PREFIX)
from oncoemotion.clinical.measure import point_e_hidden, project_scores, zscore  # noqa: E402
from oncoemotion.emotion_vectors.vectors import random_vector  # noqa: E402
from oncoemotion.clinical.baseline import NEUTRAL_BASELINE  # noqa: E402

# Confounders (everything else in the vector set is treated as an emotion).
CONFOUNDERS = ["uncertainty", "urgency", "clinical_severity", "safety_policy",
               "general_negative_valence"]
NEG_AFFECT = ["afraid_alarmed", "anxious_nervous", "sad"]

# persona -> group (order defines the spectrum layout)
SPECTRUM = [
    ("oncologo", "medici"), ("infermiere", "medici"),
    ("ingegnere", "tecnici"), ("avvocato", "tecnici"), ("contabile", "tecnici"),
    ("paziente_ansioso", "profani"), ("bambino", "profani"), ("poeta", "profani"),
    ("generico", "controlli"), ("empatico", "controlli"), ("none", "controlli"),
]



def _key_for(V, c, method, variant):
    rk = f"{c}|{method}|resid"
    if variant != "raw" and rk in V:
        return rk
    return f"{c}|{method}"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _measure(adapter, role, texts, vectors, layer_of, emo_concepts, personas):
    """Mean projection over `texts` under `role`; also the mean point-E hidden at
    each emotion best-layer (for the directional analysis)."""
    proj_acc = {c: [] for c in vectors}
    hid_acc = {c: [] for c in emo_concepts if c in vectors}
    for t in texts:
        system, user = build_decision_messages(t, role=role, personas=personas)
        ids = adapter.build_prompt_ids(user, system, assistant_prefix=TEACHER_PREFIX)
        h = point_e_hidden(adapter, ids)                    # [L+1, H]
        sc = project_scores(h, vectors, layer_of)
        for c in vectors:
            proj_acc[c].append(sc[c])
        for c in hid_acc:
            hid_acc[c].append(h[int(layer_of[c])])
    proj = {c: float(np.mean(v)) for c, v in proj_acc.items()}
    hid = {c: np.mean(np.stack(v), axis=0) for c, v in hid_acc.items()}   # [H] per concept
    return proj, hid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--limit", type=int, default=0, help="use only first N clinical stimuli")
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data/synthetic/clinical_labeled.jsonl")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/role_spectrum")
    ap.add_argument("--null-draws", type=int, default=2000,
                    help="random directions used to build the null for each cosine")
    ap.add_argument("--null-seed", type=int, default=12345)
    args = ap.parse_args()

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    best_layer = {c: val["concepts"][c]["best_layer"] for c in val["concepts"]}
    all_c = list(val["concepts"].keys())
    emo_concepts = [c for c in all_c if c not in CONFOUNDERS and _key_for(V, c, args.method, args.variant) in V]
    ctrl_concepts = [c for c in all_c if c in CONFOUNDERS and _key_for(V, c, args.method, args.variant) in V]
    concepts = emo_concepts + ctrl_concepts
    vectors = {c: V[_key_for(V, c, args.method, args.variant)] for c in concepts}
    layer_of = {c: best_layer.get(c, vectors[c].shape[0] // 2) for c in concepts}

    # clinical stimuli = EXACT items (symptoms that carry clinical + emotional content)
    items = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    stim = [it["text"] for it in items if it.get("gold_class") == "term"]
    if args.limit:
        stim = stim[:args.limit]
    print(f"{len(SPECTRUM)} personas × {len(stim)} clinical stimuli | concepts={concepts}")

    cfg = ModelConfig(dtype=args.dtype, device_map=args.device)
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()

    # Token-matched role spans, computed against THIS tokenizer: the same string
    # is a different number of tokens in each model, so a fixed pad would
    # equalize one and skew the rest. Without this everything after the system
    # block sits at a role-dependent absolute position and the role effect is
    # confounded with position; 'none' had no system block at all.
    PERSONAS, PERSONA_TOKENS = build_padded_personas(adapter.tokenizer)
    print(f'span di ruolo appaiati: {min(PERSONA_TOKENS.values())}-'
          f'{max(PERSONA_TOKENS.values())} token su {len(PERSONA_TOKENS)} ruoli', flush=True)

    # fixed reference baseline: neutral texts, NO role
    ref_proj_acc = {c: [] for c in concepts}
    for t in NEUTRAL_BASELINE:
        system, user = build_decision_messages(t, role="none", personas=PERSONAS)
        ids = adapter.build_prompt_ids(user, system, assistant_prefix=TEACHER_PREFIX)
        sc = project_scores(point_e_hidden(adapter, ids), vectors, layer_of)
        for c in concepts:
            ref_proj_acc[c].append(sc[c])
    bmean = {c: float(np.mean(ref_proj_acc[c])) for c in concepts}
    bstd = {c: float(np.std(ref_proj_acc[c]) + 1e-9) for c in concepts}

    rows = {}
    hidden_clinical = {}   # role -> {concept: [H]}
    for role, group in SPECTRUM:
        cli_proj, cli_hid = _measure(adapter, role, stim, vectors, layer_of, emo_concepts, PERSONAS)
        base_proj, _ = _measure(adapter, role, NEUTRAL_BASELINE, vectors, layer_of, emo_concepts, PERSONAS)
        zc = zscore(cli_proj, bmean, bstd)
        zb = zscore(base_proj, bmean, bstd)
        rows[role] = {
            "group": group,
            "clinical_z": {c: round(zc[c], 3) for c in concepts},
            "baseline_z": {c: round(zb[c], 3) for c in concepts},
            "reaction_z": {c: round(zc[c] - zb[c], 3) for c in concepts},
            "emo_clinical": round(float(np.mean([zc[c] for c in NEG_AFFECT if c in zc])), 3),
            "emo_baseline": round(float(np.mean([zb[c] for c in NEG_AFFECT if c in zb])), 3),
        }
        hidden_clinical[role] = cli_hid
        print(f"  {role:16} [{group:9}] emo(clinical)={rows[role]['emo_clinical']:+.2f} "
              f"emo(persona-alone)={rows[role]['emo_baseline']:+.2f}", flush=True)

    # ---- DIRECTION: is the shift aligned with the emotion axes? ----
    # per concept, grand mean of persona clinical hidden states; each persona's
    # deviation projected on the unit emotion direction at its best layer.
    direction = {}
    group_of = dict(SPECTRUM)
    for c in emo_concepts:
        u = _unit(vectors[c][int(layer_of[c])])
        H = {r: hidden_clinical[r][c] for r in hidden_clinical if c in hidden_clinical[r]}
        grand = np.mean(np.stack(list(H.values())), axis=0)
        # per-persona alignment: cosine(persona - grand, u)
        align = {}
        for r, hv in H.items():
            d = hv - grand
            nd = np.linalg.norm(d)
            align[r] = round(float(np.dot(d, u) / nd), 3) if nd > 1e-9 else 0.0
        # group-difference alignment: (profani mean - medici mean) . u  (cosine)
        def gmean(g):
            vs = [H[r] for r in H if group_of.get(r) == g]
            return np.mean(np.stack(vs), axis=0) if vs else None
        med, prof, tec = gmean("medici"), gmean("profani"), gmean("tecnici")
        def cos(a, b):
            if a is None or b is None:
                return None
            d = a - b
            nd = np.linalg.norm(d)
            return round(float(np.dot(d, u) / nd), 3) if nd > 1e-9 else 0.0
        # Null distribution: the same group-difference vector projected on random
        # unit directions of the same dimensionality. Without it a cosine has no
        # scale -- the previous read of this block called 0.2 "low" and concluded
        # the shift was not fear, but the largest cosine observed over 9 models x
        # 25 axes was 0.186, so every axis passed that threshold and the test could
        # not fail. What matters is where an axis sits against chance and against
        # the other axes, not against 1.0.
        null_cos = {}
        for pair_name, (a, b) in (("profani_minus_medici", (prof, med)),
                                  ("profani_minus_tecnici", (prof, tec)),
                                  ("tecnici_minus_medici", (tec, med))):
            if a is None or b is None:
                null_cos[pair_name] = None
                continue
            d = a - b
            nd = np.linalg.norm(d)
            if nd <= 1e-9:
                null_cos[pair_name] = None
                continue
            draws = [float(np.dot(d, random_vector(d.shape[0], seed=args.null_seed + k)) / nd)
                     for k in range(args.null_draws)]
            null_cos[pair_name] = {
                "mean": round(float(np.mean(draws)), 5),
                "sd": round(float(np.std(draws)), 5),
                "abs_p95": round(float(np.percentile(np.abs(draws), 95)), 5),
                "n_draws": args.null_draws,
            }

        # Anisotropy at this concept's extraction layer: mean pairwise cosine of
        # the raw persona states. Reported, not corrected -- the quantities above
        # are differences of means, in which an additive common component cancels
        # on both sides. This is the number Jeong (arXiv:2604.11050) asks for so a
        # reader can flag models above ~0.95.
        hv = np.stack(list(H.values())) if len(H) > 1 else None
        if hv is not None:
            hn = hv / (np.linalg.norm(hv, axis=1, keepdims=True) + 1e-9)
            gram = hn @ hn.T
            iu = np.triu_indices(len(hn), k=1)
            aniso = round(float(np.mean(gram[iu])), 4)
        else:
            aniso = None

        direction[c] = {
            "per_persona_alignment": align,
            "profani_minus_medici_cos": cos(prof, med),
            "profani_minus_tecnici_cos": cos(prof, tec),
            "tecnici_minus_medici_cos": cos(tec, med),
            "null_cos": null_cos,
            "anisotropy_at_layer": aniso,
        }

    # ---- DIRECTION SUMMARY: rank each axis, and score it against its own null ----
    # This is what the C2 section of the report should show. An absolute cosine is
    # uninterpretable on its own; a rank among the 25 axes and a z against random
    # directions are both interpretable and neither depends on a chosen threshold.
    direction_summary = {}
    for pair_name in ("profani_minus_medici", "profani_minus_tecnici", "tecnici_minus_medici"):
        key = f"{pair_name}_cos"
        vals = {c: direction[c][key] for c in emo_concepts if direction[c].get(key) is not None}
        if not vals:
            continue
        order = sorted(vals, key=lambda c: -abs(vals[c]))
        ranks = {c: i + 1 for i, c in enumerate(order)}
        med_abs = float(np.median([abs(v) for v in vals.values()]))
        per_axis = {}
        for c, v in vals.items():
            nc = direction[c].get("null_cos", {}).get(pair_name)
            z = round(v / nc["sd"], 2) if nc and nc["sd"] > 1e-12 else None
            per_axis[c] = {
                "cos": v,
                "rank": ranks[c],
                "abs_over_median": round(abs(v) / med_abs, 2) if med_abs > 1e-12 else None,
                "z_vs_random": z,
                "exceeds_null_p95": (nc is not None and abs(v) > nc["abs_p95"]),
            }
        direction_summary[pair_name] = {
            "n_axes": len(vals),
            "median_abs_cos": round(med_abs, 5),
            "max_abs_cos": round(max(abs(v) for v in vals.values()), 5),
            "ranked_axes": order,
            "per_axis": per_axis,
        }

    slug = re.sub(r"[^a-z0-9]+", "-", adapter.config.model_id.split("/")[-1].lower()).strip("-")
    out = {
        "model_id": adapter.config.model_id, "method": args.method, "variant": args.variant,
        "n_stimuli": len(stim), "concepts": concepts, "layer_of": layer_of,
        "emo_concepts": emo_concepts, "controls": ctrl_concepts,
        "spectrum": [r for r, _ in SPECTRUM], "groups": group_of,
        "personas": rows, "direction": direction,
        "direction_summary": direction_summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{slug}__spectrum.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")

    # console summary — the spectrum, sorted by clinical fear
    print("\n=== emotività (paura+ansia+tristezza) reagendo ai sintomi, per persona ===")
    order = sorted(rows, key=lambda r: rows[r]["emo_clinical"])
    for r in order:
        print(f"  {r:16} [{rows[r]['group']:9}] clinical={rows[r]['emo_clinical']:+.2f}  "
              f"persona-alone={rows[r]['emo_baseline']:+.2f}  reaction={rows[r]['emo_clinical']-rows[r]['emo_baseline']:+.2f}")
    af = direction.get("afraid_alarmed", {})
    print(f"\nDirezione (asse paura): profani−medici cos={af.get('profani_minus_medici_cos')}  "
          f"tecnici−medici cos={af.get('tecnici_minus_medici_cos')}  "
          f"profani−tecnici cos={af.get('profani_minus_tecnici_cos')}")
    print(f"\nWrote -> {args.out / f'{slug}__spectrum.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
