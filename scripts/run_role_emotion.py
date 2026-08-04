#!/usr/bin/env python
"""[Role x affect study] Does a system ROLE alter sensitivity to patient-affective
language and the resulting PRO-CTCAE code?

Factorial per dataset item (each item is already a neutral or emotional framing):
    role     in {oncologo (medical), generico (non-medical), none (baseline)}
    arm in {intact, emotion, random}  # emotion = fear/anxiety/sad directions removed
                                      # at point E; random = norm- and layer-matched
                                      # random directions removed instead (the control)

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
import hashlib
import json
import re
import subprocess
import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.clinical.baseline import NEUTRAL_BASELINE  # noqa: E402
from oncoemotion.clinical.classify import (  # noqa: E402
    build_candidates,
    build_term_matcher,
    predict_generative,
    predict_label,
)
from oncoemotion.clinical.measure import (  # noqa: E402
    decision_summary,
    hidden_at_positions,
    project_scores,
    zscore,
)
from oncoemotion.clinical.prompt import (  # noqa: E402
    TEACHER_PREFIX,
    build_decision_messages,
    build_padded_personas,
    read_point_index,
)
from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.emotion_vectors.seeds import LEXICAL_CONTROLS  # noqa: E402
from oncoemotion.emotion_vectors.vectors import random_vector  # noqa: E402
from oncoemotion.factory import build_default_mapper  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.schemas import MapRequest  # noqa: E402
from oncoemotion.steering.runtime import SteeringRuntime  # noqa: E402
from oncoemotion.terminology.pro_ctcae import load_pro_ctcae  # noqa: E402

# All concepts that are NOT confounders are treated as emotions (derived from the
# vector set, so adding emotions to seeds.py flows through automatically).
# Concepts excluded from the emotion set. The five original confounders, plus the
# two lexical controls: those exist to be measured AGAINST the emotion axes, so
# leaving them in would put them in the C2 ranking and the C5 heatmap as if they
# were emotions.
CONFOUNDERS = ["uncertainty", "urgency", "clinical_severity", "safety_policy",
               "general_negative_valence", *LEXICAL_CONTROLS]
# The causal "remove emotionality" ablation targets the clinically-relevant
# negative-affect core (kept small so the intervention is interpretable).
DEFAULT_ABLATE_CONCEPTS = ["afraid_alarmed", "anxious_nervous", "sad"]

# The z-score baseline is shared with run_role_spectrum.py so that the z-scores of
# experiments B and C are finally on the same scale (see clinical/baseline.py).


def _key_for(V, c, method, variant):
    rk = f"{c}|{method}|resid"
    if variant != "raw" and rk in V:
        return rk
    return f"{c}|{method}"


def _ablation_ctx(rt: SteeringRuntime, ablate_vecs, layer_of, arm: str = "emotion",
                  seed: int = 12345):
    """Nested ablation of each emotion direction so its projection at the readout
    layer is removed AND the change propagates to the decision.

    ``hidden_states[L]`` is the OUTPUT of block ``L-1``; the readout/best layer is
    an index into that tuple. To zero the component at readout layer ``L`` we hook
    block ``L-1`` (whose output is ``hidden_states[L]``), using the direction at
    ``L``. This is the off-by-one fix vs. hooking block ``L`` directly.

    ``arm="random"`` ablates random directions of the same norm, at the same
    layers, in the same number -- the control the ablation never had. Without it a
    flip rate of 11-38% measures "what happens if you disturb the state", not "what
    happens if you remove fear". The steering experiment already suggests the
    control matters: the emotion direction beats a random one of equal norm in
    about half the tested models, and a random direction can be more disruptive
    than the emotion direction.
    """
    stack = ExitStack()
    for i, (c, vec_LH) in enumerate(sorted(ablate_vecs.items())):
        l = int(layer_of[c])
        l = min(l, vec_LH.shape[0] - 1)
        v = vec_LH[l]
        if arm == "random":
            v = random_vector(v.shape[0], seed=seed + i, norm=float(np.linalg.norm(v)))
        hook_layer = max(0, l - 1)
        stack.enter_context(rt.intervene(hook_layer, v, 0.0, mode="ablate"))
    return stack


def _subsample_pairs(items, n_pairs, seed):
    """Keep ``n_pairs`` item pairs, stratified by gold category.

    Taking the first N in file order would take only EXACT_PRO_MATCH items -- the
    term block comes first -- so a smoke run would never touch the abstain path and
    a reduced ablation arm would silently drop the false-positive categories.
    """
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], {}).setdefault(it["pair_id"], []).append(it)
    total = len({it["pair_id"] for it in items})
    if not n_pairs or n_pairs >= total:
        return items, total
    rng = np.random.default_rng(seed)
    keep = set()
    for cat, pairs in sorted(by_cat.items()):
        ids = sorted(pairs)
        take = max(1, round(n_pairs * len(ids) / total))
        keep.update(rng.permutation(ids)[:take].tolist())
    return [it for it in items if it["pair_id"] in keep], len(keep)


def _raw_scores(adapter, system, user, vectors, layer_of, free_text=None,
                role=None, personas=None):
    """Projections at the decision point D, and at the read point R when asked.

    Both come out of ONE forward pass: reading two positions is free, running the
    model twice would double the most expensive part of the study.
    """
    ids = adapter.build_prompt_ids(user, system, assistant_prefix=TEACHER_PREFIX)
    r_idx = None
    if free_text is not None:
        r_idx = read_point_index(adapter, free_text, ids, role=role, personas=personas)
    hs = hidden_at_positions(adapter, ids, {"D": -1, "R": r_idx})
    sc_d = project_scores(hs["D"], vectors, layer_of)
    sc_r = project_scores(hs["R"], vectors, layer_of) if hs["R"] is not None else None
    return ids, sc_d, sc_r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--method", default="diff_of_means")
    ap.add_argument("--variant", default="resid", choices=["resid", "raw"])
    ap.add_argument("--roles", nargs="+",
                    default=["oncologo", "generico", "none_task", "none_filler"],
                    help="none_filler is the control: length-matched padding, no identity, "
                         "no task. none_task names the task without an identity -- what the "
                         "old 'none' accidentally was, which made 'role vs no role' read as "
                         "'identity vs task'. 'none' is literally no system block.")
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--map-floor", type=float, default=0.72,
                    help="fuzzy score floor to accept the generated term as a PRO code")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only N item pairs, stratified by category (smoke test)")
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data/synthetic/clinical_labeled.jsonl")
    ap.add_argument("--vecs", type=Path, default=_ROOT / "outputs/checkpoints/emotion_vectors.npz")
    ap.add_argument("--val-report", type=Path, default=_ROOT / "outputs/reports/vector_validation.json")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/role_emotion")
    ap.add_argument("--arms", nargs="+", default=["intact", "emotion", "random"],
                    choices=["intact", "emotion", "random"],
                    help="'random' is the norm- and layer-matched control for 'emotion'; "
                         "without it a flip rate measures disturbance, not fear removal")
    ap.add_argument("--redact-text", action="store_true",
                    help="write source_id instead of the patient text into rows.jsonl. "
                         "The rows are zipped and exported by the notebook, so with real "
                         "free text the raw strings would leave the session and land on "
                         "disk in the clear. The model still reads the text; only the "
                         "saved record is redacted.")
    ap.add_argument("--scorer", default="both",
                    choices=["constrained", "generative", "both", "adaptive"],
                    help="both (default) keeps two separate outcomes: constrained 80-way "
                          "coding for the primary endpoint and free generation for true "
                          "abstention/non-answer. Never infer abstention from the constrained "
                          "scorer, which is forced to choose a code. adaptive runs constrained "
                          "only on term golds and generative only on abstention golds; it is "
                          "the efficient real-field policy.")
    ap.add_argument("--study-config", type=Path,
                    default=_ROOT / "configs/study_esmo_2026.yaml")
    ap.add_argument("--baseline-limit", type=int, default=0,
                    help="use only the first N neutral baseline sentences (0 = all). The "
                         "baseline is re-measured per cell, so on a tiny run it dominates the "
                         "cost; keep it full for anything whose z-scores are reported.")
    ap.add_argument("--read-point", action="store_true", default=True,
                    help="also read the state at the last token of the patient text "
                         "(R), from the same forward pass as the decision point")
    ap.add_argument("--no-read-point", dest="read_point", action="store_false")
    ap.add_argument("--ablation-seed", type=int, default=12345)
    ap.add_argument("--ablation-limit", type=int, default=0,
                    help="run the ablation arms on only N item pairs (0 = all). The intact "
                         "arm always sees every item because the primary endpoint needs it; "
                         "the ablation arms feed a flip rate, which is precise long before "
                         "the full set is used.")
    ap.add_argument("--force-unvalidated-ablation", action="store_true",
                    help="smoke/debug only: run causal arms even when affect axes fail the "
                         "pre-declared AUROC/lexical gate")
    args = ap.parse_args()
    requested_arms = list(args.arms)
    study = {}
    if args.study_config.exists():
        import yaml

        study = yaml.safe_load(args.study_config.read_text(encoding="utf-8")) or {}
    ablate_concepts = list(
        study.get("mechanistic_gate", {}).get(
            "target_axes", DEFAULT_ABLATE_CONCEPTS
        )
    )
    baseline = (NEUTRAL_BASELINE[:args.baseline_limit]
                if args.baseline_limit else NEUTRAL_BASELINE)

    V = np.load(args.vecs, allow_pickle=True)
    val = json.loads(args.val_report.read_text(encoding="utf-8"))
    protocol_gate = val.get("protocol_gate") or {}
    eligible_axes = set(protocol_gate.get("eligible_axes") or [])
    required_ablation_axes = set(ablate_concepts)
    causal_requested = bool({"emotion", "random"} & set(args.arms))
    causal_axes_valid = required_ablation_axes.issubset(eligible_axes)
    if causal_requested and not causal_axes_valid and not args.force_unvalidated_ablation:
        missing = sorted(required_ablation_axes - eligible_axes)
        print(
            "[PROTOCOL GATE] causal arms skipped: axes not eligible "
            f"({', '.join(missing) or 'validation gate unavailable'}). "
            "The intact behavioral endpoint will still run.",
            flush=True,
        )
        args.arms = [arm for arm in args.arms if arm == "intact"]
    best_layer = {c: val["concepts"][c]["best_layer"] for c in val["concepts"]}
    concepts = [c for c in val["concepts"]
                if c not in CONFOUNDERS and _key_for(V, c, args.method, args.variant) in V]
    requested_controls = list(study.get("affective_profile", {}).get("controls", []))
    control_concepts = [
        c for c in requested_controls
        if c in val["concepts"] and _key_for(V, c, args.method, args.variant) in V
    ]
    measured_concepts = list(dict.fromkeys([*concepts, *control_concepts]))
    vectors = {
        c: V[_key_for(V, c, args.method, args.variant)] for c in measured_concepts
    }
    layer_of = {
        c: best_layer.get(c, vectors[c].shape[0] // 2) for c in measured_concepts
    }
    ablate_vecs = {c: vectors[c] for c in ablate_concepts if c in vectors}

    items = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(it.get("framing") == "real" for it in items) and not args.redact_text:
        ap.error("real clinical text requires --redact-text; raw fields must not be exported")
    if args.limit:
        items, _ = _subsample_pairs(items, args.limit, args.ablation_seed)
    print(f"{len(items)} items | roles={args.roles} | arms={args.arms} | "
          f"emotions={concepts} | controls={control_concepts}")

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
    unknown_roles = sorted(set(args.roles) - set(PERSONAS))
    if unknown_roles:
        raise ValueError(f"unknown role(s): {unknown_roles}")
    active_counts = {r: PERSONA_TOKENS[r] for r in args.roles if PERSONAS[r] is not None}
    spread = max(active_counts.values()) - min(active_counts.values())
    if spread > 2:
        raise ValueError(f"active role spans are not token-matched: {active_counts}")
    print(f"span di ruolo appaiati: {active_counts} (spread={spread})", flush=True)
    rt = SteeringRuntime(adapter)

    library = load_pro_ctcae()
    matcher = build_term_matcher(library)
    # Constrained scoring: the model ranks the 80 real terms instead of writing one
    # and hoping a fuzzy matcher recognises it. Free generation discarded 22-59% of
    # answers as unmappable (median ~30%), and the discarded ones were mostly
    # correct clinical Italian the surface list simply did not contain -- so the
    # accuracy metric was measuring the matcher's vocabulary, not the model.
    candidates = (build_candidates(adapter, library)
                  if args.scorer in ('constrained', 'both', 'adaptive') else None)
    if candidates:
        print(f'punteggio vincolato su {len(candidates)} termini PRO', flush=True)
    print(f"generative classifier: greedy term -> fuzzy-map to {len(library)} PRO terms "
          f"(floor={args.map_floor})")

    # deterministic mapper reference (depends only on text)
    mapper = build_default_mapper()
    mapper_ref = {}
    mapper_by_source = {}
    for it in items:
        source_key = it.get("source_id") or it["record_id"]
        if source_key not in mapper_by_source:
            r = mapper.map(MapRequest(record_id=it["record_id"], text=it["text"]))
            mids = [p.canonical_id for p in r.pro_ctcae.predictions]
            mapper_by_source[source_key] = {
                "mapper_status": r.pro_ctcae.status,
                "mapper_pro_id": mids[0] if mids else None,
                "mapper_urgent": bool(r.safety.urgent_human_review),
            }
        mapper_ref[it["record_id"]] = mapper_by_source[source_key]

    # The intact arm needs every item: it carries the primary endpoint. The ablation
    # arms feed a label flip rate, which is already precise on a few hundred
    # comparisons, so they can run on a stratified subsample without losing anything
    # -- and that is what keeps the three-arm design affordable on Colab.
    abl_items = items
    if args.ablation_limit:
        abl_items, n_kept = _subsample_pairs(items, args.ablation_limit, args.ablation_seed)
        print(f"bracci di ablazione su {n_kept} coppie ({len(abl_items)} item), "
              f"stratificate per categoria", flush=True)

    rows = []
    for role in args.roles:
        for arm in args.arms:
            ablated = arm != "intact"
            arm_items = abl_items if ablated else items
            tag = f"{role}/{arm}"
            # 1) baseline projections under this condition
            base_raw = {c: [] for c in measured_concepts}
            base_raw_r = {c: [] for c in measured_concepts}
            for txt in baseline:
                system, user = build_decision_messages(txt, role=role, personas=PERSONAS)
                with (_ablation_ctx(rt, ablate_vecs, layer_of, arm, args.ablation_seed)
                      if ablated else nullcontext()):
                    _, sc, sc_r = _raw_scores(
                        adapter, system, user, vectors, layer_of,
                        free_text=txt if args.read_point else None,
                        role=role, personas=PERSONAS)
                for c in measured_concepts:
                    base_raw[c].append(sc[c])
                    if sc_r is not None:
                        base_raw_r[c].append(sc_r[c])
            bmean = {c: float(np.mean(base_raw[c])) for c in measured_concepts}
            bstd = {c: float(np.std(base_raw[c]) + 1e-9) for c in measured_concepts}
            # R needs its own reference: it sits at a different position, so its
            # projections have a different scale and cannot share D's baseline
            have_r = all(base_raw_r[c] for c in measured_concepts)
            bmean_r = ({c: float(np.mean(base_raw_r[c])) for c in measured_concepts}
                       if have_r else None)
            bstd_r = ({c: float(np.std(base_raw_r[c]) + 1e-9) for c in measured_concepts}
                      if have_r else None)

            # 2) items
            # Identical real strings may belong to distinct validated assessments.
            # The model sees the same prompt and therefore needs only one forward
            # pass; gold-dependent correctness/rank is still computed per source row.
            measurement_cache = {}
            for it in arm_items:
                source_key = it.get("source_id") or it["record_id"]
                scoring_population = it["gold_class"] if args.scorer == "adaptive" else "all"
                cache_key = (source_key, scoring_population)
                if cache_key not in measurement_cache:
                    system, user = build_decision_messages(
                        it["text"], role=role, personas=PERSONAS)
                    with (_ablation_ctx(rt, ablate_vecs, layer_of, arm, args.ablation_seed)
                          if ablated else nullcontext()):
                        ids, raw, raw_r = _raw_scores(
                            adapter, system, user, vectors, layer_of,
                            free_text=it["text"] if args.read_point else None,
                            role=role, personas=PERSONAS)
                        pred = None
                        need_generative = (
                            args.scorer in ("generative", "both")
                            or (args.scorer == "adaptive" and it["gold_class"] == "abstain")
                        )
                        need_constrained = (
                            args.scorer in ("constrained", "both")
                            or (args.scorer == "adaptive" and it["gold_class"] == "term")
                        )
                        if need_generative:
                            pred = predict_generative(
                                adapter, ids, matcher,
                                max_new_tokens=args.max_new_tokens, floor=args.map_floor)
                        lab = (predict_label(adapter, ids, candidates)
                               if candidates and need_constrained else None)
                        dsum = decision_summary(adapter, ids)
                    z = zscore(raw, bmean, bstd)
                    z_r = (zscore(raw_r, bmean_r, bstd_r)
                           if raw_r is not None and bmean_r is not None else None)
                    measurement_cache[cache_key] = (pred, lab, dsum, z, z_r)
                else:
                    pred, lab, dsum, z, z_r = measurement_cache[cache_key]
                # the committed code: the constrained scorer when available, because it
                # cannot fail to produce a real term
                top1 = lab.top1_id if lab else pred.top1_id
                correct = (top1 == it["gold_pro_id"]) if it["gold_class"] == "term" else None
                # rank of the gold term in the model's own ranking: a near miss and a
                # wild miss are not the same event, and binary correctness hides that
                gold_rank = None
                if lab and it["gold_pro_id"]:
                    order = sorted(lab.concept_scores, key=lambda k: -lab.concept_scores[k])
                    gold_rank = (order.index(it["gold_pro_id"]) + 1
                                 if it["gold_pro_id"] in order else None)
                # A binary right/wrong throws away most of what the decision carries:
                # picking the right term with a 0.9 margin and picking it with a 0.02
                # margin count the same. The margin is continuous, comes free out of
                # the same forward pass, and gives the role x framing contrast far
                # more resolution than 112 binary items can.
                gen = ((pred.generated if pred else "") or "").strip()
                rows.append({
                    "record_id": it["record_id"], "pair_id": it["pair_id"],
                    "text": (it.get("source_id") or it["record_id"]) if args.redact_text
                            else it["text"],
                    "text_redacted": bool(args.redact_text),
                    "grade": it.get("grade"),
                    "source_id": it.get("source_id"),
                    "source_row": it.get("source_row"),
                    "source_item": it.get("source_item"),
                    "annotation_source": it.get("annotation_source"),
                    "n_words": it.get("n_words"),
                    "crosswalk_note": it.get("crosswalk_note"),
                    "framing": it["framing"], "category": it["category"],
                    "manipulation_type": it.get("manipulation_type"),
                    "affect_family": it.get("affect_family"),
                    "gold_class": it["gold_class"], "gold_pro_id": it["gold_pro_id"],
                    "gold_pro_status": it["gold_pro_status"], "urgent": it["urgent"],
                    "role": role, "ablated": ablated, "arm": arm,
                    "model_generated": (
                        None if args.redact_text else pred.term_str if pred else None
                    ),
                    "model_generated_redacted": bool(args.redact_text),
                    "model_top1_id": top1,
                    "model_top1_term": (lab.top1_term if lab else pred.top1_term),
                    "model_map_score": pred.map_score if pred else None,
                    "model_logprob": (lab.top1_score if lab else pred.logprob),
                    "model_matched": pred.matched if pred else None,
                    "scorer": "constrained" if lab else "generative",
                    # what free generation WOULD have committed to, kept as a second
                    # outcome and as the audit trail on the matcher
                    "generative_top1_id": pred.top1_id if pred else None,
                    "generative_kind": pred.kind if pred else None,
                    "generative_map_score": pred.map_score if pred else None,
                    "generative_logprob": pred.logprob if pred else None,
                    "label_margin": round(lab.margin, 5) if lab else None,
                    "label_softmax_top1": round(lab.softmax_top1, 5) if lab else None,
                    "label_entropy": round(lab.entropy, 5) if lab else None,
                    "gold_rank": gold_rank,
                    "correct": correct,
                    "z": {c: round(z[c], 3) for c in concepts},
                    "z_controls": {c: round(z[c], 3) for c in control_concepts},
                    # R = read point, last token of the patient text; D above is
                    # the decision point. Same forward pass, different position.
                    "z_read": ({c: round(z_r[c], 3) for c in concepts}
                               if z_r is not None else None),
                    "z_controls_read": (
                        {c: round(z_r[c], 3) for c in control_concepts}
                        if z_r is not None else None),
                    # --- decision profile, all free from the same forward pass ---
                    "decision_entropy": round(dsum["entropy"], 3),
                    "decision_margin": round(dsum["top1_top2_margin"], 5),
                    "decision_top1_prob": round(dsum["top1_prob"], 5),
                    "decision_top_token_ids": dsum["top_token_ids"],
                    # abstaining on purpose and emitting something unmappable both
                    # end with top1_id None, but they are different behaviours: only
                    # the first is the model declining to code
                    "abstained": (pred.kind == "abstained") if pred else None,
                    "unmappable": (pred.kind == "unmapped") if pred else None,
                    "non_answer": (pred.kind == "non_answer") if pred else None,
                    "n_generated_tokens": len(gen.split()) if pred else None,
                    **mapper_ref[it["record_id"]],
                })
            done = sum(1 for r in rows if r["role"] == role and r["arm"] == arm)
            print(f"  [{tag:20}] {done} items measured", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", adapter.config.model_id.split("/")[-1].lower()).strip("-")
    outp = args.out / f"{slug}__rows.jsonl"
    rows_tmp = outp.with_suffix(outp.suffix + ".tmp")
    with rows_tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows_tmp.replace(outp)

    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        git_commit = None
    meta = {
        "protocol_id": study.get("protocol_id"),
        "study_config_sha256": _sha256(args.study_config),
        "git_commit": git_commit,
        "dataset_sha256": _sha256(args.dataset),
        "vectors_sha256": _sha256(args.vecs),
        "validation_sha256": _sha256(args.val_report),
        "model_id": adapter.config.model_id, "method": args.method, "variant": args.variant,
        "roles": args.roles, "concepts": concepts,
        "control_concepts": control_concepts, "layer_of": layer_of,
        "ablate_concepts": list(ablate_vecs.keys()),
        "ablate_concepts_requested": ablate_concepts,
        "n_items": len(items), "n_rows": len(rows),
        "n_unique_source_texts": len({it.get("source_id") or it["record_id"] for it in items}),
        "arms": args.arms, "ablation_seed": args.ablation_seed,
        "arms_requested": requested_arms,
        "causal_axes_valid": causal_axes_valid,
        "eligible_affect_axes": sorted(eligible_axes),
        "ablation_limit": args.ablation_limit or None,
        "n_items_ablation_arms": len(abl_items),
        "scorer": args.scorer,
        "role_token_counts": {r: PERSONA_TOKENS[r] for r in args.roles},
        "text_redacted": bool(args.redact_text),
        "rows_sha256": _sha256(outp),
    }
    meta_path = args.out / f"{slug}__meta.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)
    print(f"\nWrote {len(rows)} rows -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
