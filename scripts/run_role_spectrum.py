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
from oncoemotion.clinical.prompt import build_decision_messages, TEACHER_PREFIX  # noqa: E402
from oncoemotion.clinical.measure import point_e_hidden, project_scores, zscore  # noqa: E402

EMOTIONS = ["afraid_alarmed", "anxious_nervous", "sad", "calm", "compassionate"]
CONTROLS = ["clinical_severity", "general_negative_valence", "safety_policy", "urgency"]
NEG_AFFECT = ["afraid_alarmed", "anxious_nervous", "sad"]

# persona -> group (order defines the spectrum layout)
SPECTRUM = [
    ("oncologo", "medici"), ("infermiere", "medici"),
    ("ingegnere", "tecnici"), ("avvocato", "tecnici"), ("contabile", "tecnici"),
    ("paziente_ansioso", "profani"), ("bambino", "profani"), ("poeta", "profani"),
    ("generico", "controlli"), ("empatico", "controlli"), ("none", "controlli"),
]

NEUTRAL_BASELINE = [
    "Il modulo è stato compilato correttamente.",
    "La procedura di registrazione è terminata.",
    "Il documento è stato archiviato negli atti.",
    "L'appuntamento è confermato per la data prevista.",
    "I dati anagrafici risultano aggiornati.",
    "La pratica è stata protocollata questa mattina.",
    "Il questionario contiene dieci domande in totale.",
    "La sala d'attesa è al primo piano dell'edificio.",
]


def _key_for(V, c, method, variant):
    rk = f"{c}|{method}|resid"
    if variant != "raw" and rk in V:
        return rk
    return f"{c}|{method}"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _measure(adapter, role, texts, vectors, layer_of):
    """Mean projection over `texts` under `role`; also the mean point-E hidden at
    each emotion best-layer (for the directional analysis)."""
    proj_acc = {c: [] for c in vectors}
    hid_acc = {c: [] for c in EMOTIONS if c in vectors}
    for t in texts:
        system, user = build_decision_messages(t, role=role)
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
    args = ap.parse_args()

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    best_layer = {c: val["concepts"][c]["best_layer"] for c in val["concepts"]}
    concepts = [c for c in (EMOTIONS + CONTROLS) if _key_for(V, c, args.method, args.variant) in V]
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

    # fixed reference baseline: neutral texts, NO role
    ref_proj_acc = {c: [] for c in concepts}
    for t in NEUTRAL_BASELINE:
        system, user = build_decision_messages(t, role="none")
        ids = adapter.build_prompt_ids(user, system, assistant_prefix=TEACHER_PREFIX)
        sc = project_scores(point_e_hidden(adapter, ids), vectors, layer_of)
        for c in concepts:
            ref_proj_acc[c].append(sc[c])
    bmean = {c: float(np.mean(ref_proj_acc[c])) for c in concepts}
    bstd = {c: float(np.std(ref_proj_acc[c]) + 1e-9) for c in concepts}

    rows = {}
    hidden_clinical = {}   # role -> {concept: [H]}
    for role, group in SPECTRUM:
        cli_proj, cli_hid = _measure(adapter, role, stim, vectors, layer_of)
        base_proj, _ = _measure(adapter, role, NEUTRAL_BASELINE, vectors, layer_of)
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
    for c in [e for e in EMOTIONS if e in vectors]:
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
        direction[c] = {
            "per_persona_alignment": align,
            "profani_minus_medici_cos": cos(prof, med),
            "profani_minus_tecnici_cos": cos(prof, tec),
            "tecnici_minus_medici_cos": cos(tec, med),
        }

    slug = re.sub(r"[^a-z0-9]+", "-", adapter.config.model_id.split("/")[-1].lower()).strip("-")
    out = {
        "model_id": adapter.config.model_id, "method": args.method, "variant": args.variant,
        "n_stimuli": len(stim), "concepts": concepts, "layer_of": layer_of,
        "spectrum": [r for r, _ in SPECTRUM], "groups": group_of,
        "personas": rows, "direction": direction,
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
