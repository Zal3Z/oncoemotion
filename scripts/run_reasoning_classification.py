#!/usr/bin/env python
"""Run the joint ontology + explicit-abstention reasoning extension for one model.

This is deliberately separate from the frozen real-field v1 endpoint.  Each
validated row is classified into one of three explicit alternatives:

* ``PRO-CTCAE | <item>``
* ``CTCAE | <item>``
* ``NON_CLASSIFICABILE``

The direct condition scores the choices immediately.  The portable deliberative
condition first asks every model for the same short evidence/uncertainty note with
native thinking disabled.  A separately configured ``native_reasoning`` condition
uses a model's documented thinking switch and is never pooled with the portable
condition.  Generated notes are never retained in redacted result packages; only a
hash and token count remain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.clinical.classify import predict_label  # noqa: E402
from oncoemotion.clinical.joint import (  # noqa: E402
    NON_CLASSIFICABILE_ID,
    build_joint_candidates,
    decode_joint_choice,
    gold_joint_choice,
    redacted_reasoning_digest,
)
from oncoemotion.clinical.measure import decision_summary  # noqa: E402
from oncoemotion.clinical.prompt import (  # noqa: E402
    JOINT_TEACHER_PREFIX,
    build_decision_messages,
    build_deliberation_messages,
    build_padded_personas,
)
from oncoemotion.config import ModelConfig  # noqa: E402
from oncoemotion.models.base import load_adapter  # noqa: E402
from oncoemotion.terminology.pro_ctcae import load_pro_ctcae  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _generate_deliberation(
    adapter,
    system: str | None,
    user: str,
    max_new_tokens: int,
    *,
    native_thinking: bool,
    native_sampling: dict | None = None,
) -> str:
    """Generate a portable note or a model-native thinking trace."""
    import torch

    tok = adapter.tokenizer
    ids = adapter.build_prompt_ids(
        user,
        system,
        chat_template_kwargs={"enable_thinking": native_thinking},
    )
    generation = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": (
            tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ),
    }
    if native_thinking:
        sampling = dict(native_sampling or {})
        generation.update(
            {
                "do_sample": bool(sampling.get("do_sample", True)),
                "temperature": float(sampling.get("temperature", 0.6)),
                "top_p": float(sampling.get("top_p", 0.95)),
                "top_k": int(sampling.get("top_k", 20)),
            }
        )
    else:
        generation.update(
            {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
            }
        )
    with torch.no_grad():
        output = adapter.model.generate(ids, **generation)
    generated = output[0][ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True).strip()


def _reasoning_tokens(adapter, text: str) -> int:
    if not text:
        return 0
    return len(adapter.tokenizer(text, add_special_tokens=False).input_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quantization", choices=["nf4", "int8"])
    ap.add_argument("--dataset", type=Path, default=_ROOT / "data/real/clinical_real.jsonl")
    ap.add_argument("--out", type=Path, default=_ROOT / "outputs/reasoning_real")
    ap.add_argument(
        "--study-config",
        type=Path,
        default=_ROOT / "configs/study_esmo_2026_reasoning.yaml",
    )
    ap.add_argument("--roles", nargs="+", default=["oncologo", "none_filler"])
    ap.add_argument(
        "--reasoning-modes",
        nargs="+",
        default=["direct", "deliberative"],
        choices=["direct", "deliberative", "native_reasoning"],
    )
    ap.add_argument("--reasoning-max-new-tokens", type=int, default=80)
    ap.add_argument("--native-reasoning-max-new-tokens", type=int, default=0)
    ap.add_argument("--candidate-chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redact-text", action="store_true")
    args = ap.parse_args()

    import yaml

    study = yaml.safe_load(args.study_config.read_text(encoding="utf-8")) or {}
    native_config = study.get("native_reasoning", {}) or {}
    native_models = set(native_config.get("models", []))
    if "native_reasoning" in args.reasoning_modes and args.model not in native_models:
        ap.error(
            f"native_reasoning is not declared for {args.model}; configured: "
            f"{sorted(native_models)}"
        )
    native_max = int(
        args.native_reasoning_max_new_tokens
        or native_config.get("max_new_tokens", 512)
    )
    native_sampling = dict(native_config.get("sampling", {}) or {})
    items = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        # Deterministic pilot preserving source order.  The definitive run is full.
        items = items[: args.limit]
    if any(item.get("framing") == "real" for item in items) and not args.redact_text:
        ap.error("real clinical text requires --redact-text")

    cfg = ModelConfig(
        dtype=args.dtype,
        device_map=args.device,
        quantization=args.quantization,
    )
    adapter = load_adapter(args.model, cfg)
    print(f"Loading {adapter.config.model_id} ...", flush=True)
    adapter.load()

    personas, persona_tokens = build_padded_personas(adapter.tokenizer)
    unknown_roles = sorted(set(args.roles) - set(personas))
    if unknown_roles:
        raise ValueError(f"unknown role(s): {unknown_roles}")

    pro_library = load_pro_ctcae()
    ctcae_terms = sorted(
        {str(item["gold_ctcae_term"]).strip() for item in items if item.get("gold_ctcae_term")}
    )
    candidates = build_joint_candidates(adapter, pro_library, ctcae_terms)
    candidate_by_id = {candidate.canonical_id: candidate for candidate in candidates}
    print(
        f"{len(items)} rows | {len(candidates)} joint choices "
        f"({len(pro_library)} PRO + {len(ctcae_terms)} CTCAE + explicit abstention)",
        flush=True,
    )

    rows: list[dict] = []
    for role in args.roles:
        for reasoning_mode in args.reasoning_modes:
            cache: dict[str, tuple] = {}
            for item in items:
                source_key = str(item.get("source_id") or item["record_id"])
                if source_key not in cache:
                    deliberation = ""
                    if reasoning_mode in {"deliberative", "native_reasoning"}:
                        r_system, r_user = build_deliberation_messages(
                            item["text"],
                            role=role,
                            personas=personas,
                        )
                        native_thinking = reasoning_mode == "native_reasoning"
                        deliberation = _generate_deliberation(
                            adapter,
                            r_system,
                            r_user,
                            native_max if native_thinking else args.reasoning_max_new_tokens,
                            native_thinking=native_thinking,
                            native_sampling=native_sampling,
                        )
                    system, user = build_decision_messages(
                        item["text"],
                        role=role,
                        personas=personas,
                        decision_space="joint",
                        deliberation=deliberation or None,
                    )
                    # Disable native thinking in the final constrained decision.
                    # The only difference between modes is the auditable first pass.
                    ids = adapter.build_prompt_ids(
                        user,
                        system,
                        assistant_prefix=JOINT_TEACHER_PREFIX,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                    prediction = predict_label(
                        adapter,
                        ids,
                        candidates,
                        chunk=args.candidate_chunk,
                        top_k=5,
                    )
                    cache[source_key] = (
                        deliberation,
                        prediction,
                        decision_summary(adapter, ids),
                    )
                deliberation, prediction, dsum = cache[source_key]

                display = candidate_by_id[prediction.top1_id].term_en
                predicted = decode_joint_choice(prediction.top1_id, display)
                gold = gold_joint_choice(item)
                strict_correct = predicted.choice_id == gold.choice_id
                top5 = [
                    {"choice_id": cid, "item": term, "score": round(float(score), 6)}
                    for cid, term, score in prediction.ranking
                ]
                rows.append(
                    {
                        "record_id": item["record_id"],
                        "pair_id": item.get("pair_id"),
                        "source_id": item.get("source_id"),
                        "source_row": item.get("source_row"),
                        "text": source_key if args.redact_text else item["text"],
                        "text_redacted": bool(args.redact_text),
                        "n_words": item.get("n_words"),
                        "grade": item.get("grade"),
                        "annotation_source": item.get("annotation_source"),
                        "source_item": item.get("source_item"),
                        "gold_choice_id": gold.choice_id,
                        "gold_system": gold.system,
                        "gold_item": gold.item,
                        "gold_pro_id": item.get("gold_pro_id"),
                        "gold_ctcae_term": item.get("gold_ctcae_term"),
                        "role": role,
                        "reasoning_mode": reasoning_mode,
                        "reasoning_backend": {
                            "direct": "none",
                            "deliberative": "standardized_prompted",
                            "native_reasoning": "native_chat_template_thinking",
                        }[reasoning_mode],
                        "native_thinking_enabled": reasoning_mode == "native_reasoning",
                        "decision_space": "joint",
                        "model_choice_id": predicted.choice_id,
                        "model_choice_system": predicted.system,
                        "model_choice_item": predicted.item,
                        "model_top1_id": predicted.pro_id,
                        "model_top1_term": predicted.item,
                        "correct": strict_correct,
                        "system_correct": predicted.system == gold.system,
                        "pro_correct": (
                            strict_correct if gold.system == "PRO-CTCAE" else None
                        ),
                        "ctcae_correct": strict_correct if gold.system == "CTCAE" else None,
                        "nonclassifiable_correct": (
                            strict_correct if gold.system == "NON_CLASSIFICABILE" else None
                        ),
                        "explicit_nonclassifiable": (
                            predicted.choice_id == NON_CLASSIFICABILE_ID
                        ),
                        "label_margin": round(prediction.margin, 6),
                        "label_softmax_top1": round(prediction.softmax_top1, 6),
                        "label_entropy": round(prediction.entropy, 6),
                        "top5": top5,
                        "reasoning_text": (
                            None if args.redact_text else deliberation or None
                        ),
                        "reasoning_generated_redacted": bool(args.redact_text),
                        "reasoning_sha256_16": redacted_reasoning_digest(deliberation),
                        "reasoning_n_tokens": _reasoning_tokens(adapter, deliberation),
                        "decision_entropy": round(dsum["entropy"], 6),
                        "decision_margin": round(dsum["top1_top2_margin"], 6),
                        "decision_top1_prob": round(dsum["top1_prob"], 6),
                    }
                )
            print(
                f"  [{role}/{reasoning_mode}] {len(items)} assessment rows; "
                f"{len(cache)} unique texts",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    slug = _slug(adapter.config.model_id)
    rows_path = args.out / f"{slug}__rows.jsonl"
    tmp = rows_path.with_suffix(rows_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(rows_path)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            text=True,
        ).strip()
    except Exception:
        git_commit = None
    meta = {
        "protocol_id": study.get("protocol_id"),
        "study_config_sha256": _sha256(args.study_config),
        "dataset_sha256": _sha256(args.dataset),
        "rows_sha256": _sha256(rows_path),
        "git_commit": git_commit,
        "model_id": adapter.config.model_id,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "roles": args.roles,
        "reasoning_modes": args.reasoning_modes,
        "decision_space": "joint",
        "explicit_nonclassifiable": True,
        "n_pro_candidates": len(pro_library),
        "n_ctcae_candidates": len(ctcae_terms),
        "n_candidates": len(candidates),
        "n_items": len(items),
        "n_unique_source_texts": len(
            {item.get("source_id") or item["record_id"] for item in items}
        ),
        "n_rows": len(rows),
        "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
        "native_reasoning_max_new_tokens": native_max,
        "native_reasoning_sampling": native_sampling,
        "candidate_chunk": args.candidate_chunk,
        "role_token_counts": {role: persona_tokens[role] for role in args.roles},
        "text_redacted": bool(args.redact_text),
        "reasoning_text_redacted": bool(args.redact_text),
    }
    meta_path = args.out / f"{slug}__meta.json"
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_tmp.replace(meta_path)
    print(f"Wrote {len(rows)} rows -> {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
