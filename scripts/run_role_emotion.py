#!/usr/bin/env python
"""[Role x Emotion study] Does the assigned ROLE change the model's emotionality,
and does emotionality change how it LABELS PRO-CTCAE (right vs wrong)?

Factorial per dataset item (each item is already a neutral or emotional framing):
    role     in {oncologo (medical), generico (non-medical), none (baseline)}
    ablation in {intact, ablated}   # ablated = emotion directions removed at point E

For every (item, role, ablation) we record, at the identical teacher-forced point E:
  * emotion-like z-scores (projection onto Phase-2 emotion vectors vs a neutral
    baseline measured under the same condition);
  * the MODEL's committed PRO term (constrained scoring of the 80 terms) + confidence;
  * next-token decision entropy;
  * the deterministic MAPPER's label (reference; depends only on the text).

Correctness: for EXACT (term) golds, correct = model top-1 == gold term. For
abstain golds (negated / no-direct / out-of-scope / insufficient / urgent) there is
no correct term; we log the model's confidence to study *false-positive coding*.

Usage:
    python scripts/run_role_emotion.py --limit 4          # quick local smoke test
    python scripts/run_role_emotion.py --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.clinical.prompt import build_decision_messages, TEACHER_PREFIX  # noqa: E402
from oncoemotion.clinical.measure import point_e_hidden, project_scores, zscore, decision_summary  # noqa: E402
from oncoemotion.clinical.classify import build_term_matcher, predict_generative  # noqa: E402
from oncoemotion.steering.runtime import SteeringRuntime  # noqa: E402
from oncoemotion.terminology.pro_ctcae import load_pro_ctcae  # noqa: E402
from oncoemotion.factory import build_default_mapper  # noqa: E402
from oncoemotion.schemas import MapRequest  # noqa: E402

# All concepts that are NOT confounders are treated as emotions (derived from the
# vector set, so adding emotions to seeds.py flows through automatically).
CONFOUNDERS = ["uncertainty", "urgency", "clinical_severity", "safety_policy",
               "general_negative_valence"]
# The causal "remove emotionality" ablation targets the clinically-relevant
# negative-affect core (kept small so the intervention is interpretable).
ABLATE_CONCEPTS = ["afraid_alarmed", "anxious_nervous", "sad"]

# Emotionally-neutral, non-clinical routine sentences for the z-score baseline.
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


def _ablation_ctx(rt: SteeringRuntime, ablate_vecs, layer_of):
    """Nested ablation of each emotion direction so its projection at the readout
    layer is removed AND the change propagates to the decision.

    ``hidden_states[L]`` is the OUTPUT of block ``L-1``; the readout/best layer is
    an index into that tuple. To zero the component at readout layer ``L`` we hook
    block ``L-1`` (whose output is ``hidden_states[L]``), using the direction at
    ``L``. This is the off-by-one fix vs. hooking block ``L`` directly.
    """
    stack = ExitStack()
    for c, vec_LH in ablate_vecs.items():
        l = int(layer_of[c])
        l = min(l, vec_LH.shape[0] - 1)
        hook_layer = max(0, l - 1)
        stack.enter_context(rt.intervene(hook_layer, vec_LH[l], 0.0, mode="ablate"))
    return stack


def _raw_scores(adapter, system, user, vectors, layer_of):
    ids = adapter.build_prompt_ids(user, system, assistant_prefix=TEACHER_PREFIX)
    h = point_e_hidden(adapter, ids)
    return ids, project_scores(h, vectors, layer_of)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--roles", nargs="+", default=["oncologo", "generico", "none"])
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--map-floor", type=float, default=0.72,
                    help="fuzzy score floor to accept the generated term as a PRO code")
    ap.add_argument("--limit", type=int, default=0, help="only first N pairs (smoke test)")
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data/synthetic/clinical_labeled.jsonl")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/role_emotion")
    args = ap.parse_args()

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    best_layer = {c: val["concepts"][c]["best_layer"] for c in val["concepts"]}
    concepts = [c for c in val["concepts"]
                if c not in CONFOUNDERS and _key_for(V, c, args.method, args.variant) in V]
    vectors = {c: V[_key_for(V, c, args.method, args.variant)] for c in concepts}
    layer_of = {c: best_layer.get(c, vectors[c].shape[0] // 2) for c in concepts}
    ablate_vecs = {c: vectors[c] for c in ABLATE_CONCEPTS if c in vectors}

    items = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        pairs = []
        seen = set()
        for it in items:
            if it["pair_id"] not in seen:
                seen.add(it["pair_id"])
            if len(seen) > args.limit:
                break
            pairs.append(it)
        items = pairs
    print(f"{len(items)} items | roles={args.roles} | ablation=[intact,ablated] | concepts={concepts}")

    cfg = ModelConfig(dtype=args.dtype, device_map=args.device)
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()
    rt = SteeringRuntime(adapter)

    library = load_pro_ctcae()
    matcher = build_term_matcher(library)
    print(f"generative classifier: greedy term -> fuzzy-map to {len(library)} PRO terms "
          f"(floor={args.map_floor})")

    # deterministic mapper reference (depends only on text)
    mapper = build_default_mapper()
    mapper_ref = {}
    for it in items:
        r = mapper.map(MapRequest(record_id=it["record_id"], text=it["text"]))
        mids = [p.canonical_id for p in r.pro_ctcae.predictions]
        mapper_ref[it["record_id"]] = {
            "mapper_status": r.pro_ctcae.status,
            "mapper_pro_id": mids[0] if mids else None,
            "mapper_urgent": bool(r.safety.urgent_human_review),
        }

    rows = []
    for role in args.roles:
        for ablated in (False, True):
            tag = f"{role}/{'ablated' if ablated else 'intact'}"
            # 1) baseline projections under this condition
            base_raw = {c: [] for c in concepts}
            for txt in NEUTRAL_BASELINE:
                system, user = build_decision_messages(txt, role=role)
                with (_ablation_ctx(rt, ablate_vecs, layer_of) if ablated else nullcontext()):
                    _, sc = _raw_scores(adapter, system, user, vectors, layer_of)
                for c in concepts:
                    base_raw[c].append(sc[c])
            bmean = {c: float(np.mean(base_raw[c])) for c in concepts}
            bstd = {c: float(np.std(base_raw[c]) + 1e-9) for c in concepts}

            # 2) items
            for it in items:
                system, user = build_decision_messages(it["text"], role=role)
                with (_ablation_ctx(rt, ablate_vecs, layer_of) if ablated else nullcontext()):
                    ids, raw = _raw_scores(adapter, system, user, vectors, layer_of)
                    pred = predict_generative(adapter, ids, matcher,
                                              max_new_tokens=args.max_new_tokens, floor=args.map_floor)
                    dsum = decision_summary(adapter, ids)
                z = zscore(raw, bmean, bstd)
                correct = (pred.top1_id == it["gold_pro_id"]) if it["gold_class"] == "term" else None
                rows.append({
                    "record_id": it["record_id"], "pair_id": it["pair_id"],
                    "text": it["text"],
                    "framing": it["framing"], "category": it["category"],
                    "gold_class": it["gold_class"], "gold_pro_id": it["gold_pro_id"],
                    "gold_pro_status": it["gold_pro_status"], "urgent": it["urgent"],
                    "role": role, "ablated": ablated,
                    "model_generated": pred.term_str,
                    "model_top1_id": pred.top1_id, "model_top1_term": pred.top1_term,
                    "model_map_score": pred.map_score,
                    "model_logprob": pred.logprob,
                    "model_matched": pred.matched,
                    "correct": correct,
                    "z": {c: round(z[c], 3) for c in concepts},
                    "decision_entropy": round(dsum["entropy"], 3),
                    **mapper_ref[it["record_id"]],
                })
            done = sum(1 for r in rows if r["role"] == role and r["ablated"] == ablated)
            print(f"  [{tag:20}] {done} items measured", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", adapter.config.model_id.split("/")[-1].lower()).strip("-")
    outp = args.out / f"{slug}__rows.jsonl"
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "model_id": adapter.config.model_id, "method": args.method, "variant": args.variant,
        "roles": args.roles, "concepts": concepts, "layer_of": layer_of,
        "ablate_concepts": list(ablate_vecs.keys()),
        "n_items": len(items), "n_rows": len(rows),
    }
    (args.out / f"{slug}__meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(f"\nWrote {len(rows)} rows -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
