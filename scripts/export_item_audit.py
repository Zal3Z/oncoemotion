#!/usr/bin/env python
"""Build privacy-local item-level audit payloads from real-field model results.

The output is JSON/JSONL rather than an exported workbook so the scientific data
lineage remains independent of any spreadsheet library.  A workbook renderer can
consume the payload without reopening model artifacts.  Raw clinical text is read
only from the local ingested dataset and never from the redacted result package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _model_label(meta: dict, fallback: str) -> str:
    return str(meta.get("model_id") or fallback)


def _current_evaluation(row: dict) -> tuple[str, bool | None]:
    source = row.get("annotation_source")
    if source == "PRO-CTCAE":
        return "PRO_ITEM", bool(row.get("correct"))
    if source == "CTCAE v5":
        # v1 offered no CTCAE candidate: this row can audit false PRO coding or
        # abstention, but cannot establish CTCAE item accuracy.
        return "CTCAE_ITEM_NOT_OFFERED_IN_V1", None
    if source == "Non associabile":
        return "EXPECTED_ABSTENTION", row.get("generative_kind") == "abstained"
    return "UNRECOGNISED_SOURCE", None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path.home() / "Downloads/oncoemotion_real_results/outputs/real_fields",
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/real/clinical_real.jsonl",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/tables/item_audit",
    )
    ap.add_argument(
        "--reasoning-results-dir",
        type=Path,
        help="optional joint direct/deliberative results to join to the same source",
    )
    ap.add_argument("--wide-role", default="oncologo")
    args = ap.parse_args()

    source_rows = _read_jsonl(args.dataset)
    source_by_record = {row["record_id"]: row for row in source_rows}
    if len(source_by_record) != len(source_rows):
        raise ValueError("duplicate record_id in local real dataset")

    row_files = sorted(args.results_dir.glob("*__rows.jsonl"))
    if not row_files:
        raise FileNotFoundError(f"no model rows under {args.results_dir}")
    long_rows: list[dict] = []
    models: list[str] = []
    dataset_hash_matches: dict[str, bool] = {}
    local_dataset_hash = _sha256(args.dataset)
    for path in row_files:
        slug = path.name.split("__")[0]
        meta_path = path.with_name(f"{slug}__meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dataset_hash_matches[slug] = meta.get("dataset_sha256") == local_dataset_hash
        model = _model_label(meta, slug)
        models.append(model)
        for result in _read_jsonl(path):
            source = source_by_record.get(result["record_id"])
            if source is None:
                raise ValueError(f"result record missing from local dataset: {result['record_id']}")
            identity_fields = (
                "source_id",
                "source_row",
                "source_item",
                "annotation_source",
                "grade",
                "n_words",
                "gold_class",
                "gold_pro_id",
                "gold_pro_status",
            )
            mismatched = [
                field for field in identity_fields if result.get(field) != source.get(field)
            ]
            if mismatched:
                raise ValueError(
                    f"local dataset differs from {slug}/{result['record_id']}: {mismatched}"
                )
            endpoint, auditable_correct = _current_evaluation(result)
            long_rows.append(
                {
                    "record_id": result["record_id"],
                    "source_id": result.get("source_id"),
                    "source_row": source.get("source_row"),
                    "text": source.get("text"),
                    "n_words": source.get("n_words"),
                    "annotation_source": source.get("annotation_source"),
                    "gold_source_item": source.get("source_item"),
                    "gold_grade": source.get("grade"),
                    "gold_pro_id": source.get("gold_pro_id"),
                    "gold_pro_term": source.get("gold_pro_term"),
                    "gold_ctcae_term": source.get("gold_ctcae_term"),
                    "model": model,
                    "model_slug": slug,
                    "role": result.get("role"),
                    "arm": result.get("arm"),
                    "model_top1_id": result.get("model_top1_id"),
                    "model_top1_term": result.get("model_top1_term"),
                    "generative_top1_id": result.get("generative_top1_id"),
                    "generative_kind": result.get("generative_kind"),
                    "gold_rank": result.get("gold_rank"),
                    "label_margin": result.get("label_margin"),
                    "label_softmax_top1": result.get("label_softmax_top1"),
                    "endpoint_available_in_v1": endpoint,
                    "auditable_correct_v1": auditable_correct,
                    "manual_expected_system": None,
                    "manual_expected_item": None,
                    "manual_correct_override": None,
                    "manual_note": None,
                }
            )

    # One row per validated assessment, with a compact nested block for every
    # model in the requested role.  This is the natural source for a wide Excel tab.
    wide: dict[str, dict] = {}
    for row in long_rows:
        if row["role"] != args.wide_role or row["arm"] != "intact":
            continue
        record = wide.setdefault(
            row["record_id"],
            {
                key: row[key]
                for key in (
                    "record_id",
                    "source_id",
                    "source_row",
                    "text",
                    "n_words",
                    "annotation_source",
                    "gold_source_item",
                    "gold_grade",
                    "gold_pro_id",
                    "gold_pro_term",
                    "gold_ctcae_term",
                )
            },
        )
        record.setdefault("models", {})[row["model"]] = {
            "model_top1_id": row["model_top1_id"],
            "model_top1_term": row["model_top1_term"],
            "generative_top1_id": row["generative_top1_id"],
            "generative_kind": row["generative_kind"],
            "gold_rank": row["gold_rank"],
            "label_margin": row["label_margin"],
            "auditable_correct_v1": row["auditable_correct_v1"],
            "endpoint_available_in_v1": row["endpoint_available_in_v1"],
        }

    args.outdir.mkdir(parents=True, exist_ok=True)
    long_path = args.outdir / "item_audit_long.jsonl"
    wide_path = args.outdir / "item_audit_wide.jsonl"
    with long_path.open("w", encoding="utf-8") as handle:
        for row in long_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with wide_path.open("w", encoding="utf-8") as handle:
        for row in sorted(wide.values(), key=lambda value: int(value["source_row"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["annotation_source"] for row in source_rows)
    per_model_role = defaultdict(Counter)
    for row in long_rows:
        key = f"{row['model']} | {row['role']}"
        per_model_role[key]["rows"] += 1
        if row["auditable_correct_v1"] is True:
            per_model_role[key]["auditable_correct"] += 1
        if row["auditable_correct_v1"] is not None:
            per_model_role[key]["auditable_denominator"] += 1
    summary = {
        "schema_version": "oncoemotion_item_audit/1.0",
        "privacy": "LOCAL ONLY: contains validated clinical free text",
        "source_dataset": str(args.dataset),
        "source_dataset_sha256": local_dataset_hash,
        "result_dataset_hash_matches": dataset_hash_matches,
        "join_validation": (
            "Every result row matched the local dataset on source_id, source row, "
            "validated source/item, grade, word count, gold class and gold PRO id."
        ),
        "models": models,
        "wide_role": args.wide_role,
        "n_assessments": len(source_rows),
        "n_long_rows": len(long_rows),
        "annotation_source_counts": dict(counts),
        "v1_limitation": (
            "CTCAE items were not offered to v1 models, so CTCAE item correctness "
            "cannot be calculated retrospectively from those outputs."
        ),
        "per_model_role": {key: dict(value) for key, value in per_model_role.items()},
        "files": {
            "long": str(long_path),
            "wide": str(wide_path),
        },
    }
    summary_path = args.outdir / "item_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(long_rows)} long rows -> {long_path}")
    print(f"{len(wide)} wide assessment rows -> {wide_path}")

    if args.reasoning_results_dir:
        reasoning_rows: list[dict] = []
        reasoning_models: list[str] = []
        reasoning_hash_matches: dict[str, bool] = {}
        reasoning_files = sorted(args.reasoning_results_dir.glob("*__rows.jsonl"))
        if not reasoning_files:
            raise FileNotFoundError(
                f"no reasoning model rows under {args.reasoning_results_dir}"
            )
        for path in reasoning_files:
            slug = path.name.split("__")[0]
            meta = json.loads(
                path.with_name(f"{slug}__meta.json").read_text(encoding="utf-8")
            )
            reasoning_hash_matches[slug] = meta.get("dataset_sha256") == local_dataset_hash
            model = _model_label(meta, slug)
            reasoning_models.append(model)
            for result in _read_jsonl(path):
                source = source_by_record.get(result["record_id"])
                if source is None:
                    raise ValueError(
                        f"reasoning record missing from local dataset: {result['record_id']}"
                    )
                identity_fields = (
                    "source_id",
                    "source_row",
                    "source_item",
                    "annotation_source",
                    "grade",
                    "n_words",
                    "gold_pro_id",
                    "gold_ctcae_term",
                )
                mismatched = [
                    field
                    for field in identity_fields
                    if result.get(field) != source.get(field)
                ]
                if mismatched:
                    raise ValueError(
                        "local dataset differs from reasoning result "
                        f"{slug}/{result['record_id']}: {mismatched}"
                    )
                reasoning_rows.append(
                    {
                        "record_id": result["record_id"],
                        "source_id": source.get("source_id"),
                        "source_row": source.get("source_row"),
                        "text": source.get("text"),
                        "n_words": source.get("n_words"),
                        "annotation_source": source.get("annotation_source"),
                        "gold_source_item": source.get("source_item"),
                        "gold_grade": source.get("grade"),
                        "gold_choice_id": result.get("gold_choice_id"),
                        "gold_system": result.get("gold_system"),
                        "gold_item": result.get("gold_item"),
                        "model": model,
                        "model_slug": slug,
                        "role": result.get("role"),
                        "reasoning_mode": result.get("reasoning_mode"),
                        "reasoning_backend": result.get("reasoning_backend"),
                        "native_thinking_enabled": result.get(
                            "native_thinking_enabled"
                        ),
                        "model_choice_id": result.get("model_choice_id"),
                        "model_choice_system": result.get("model_choice_system"),
                        "model_choice_item": result.get("model_choice_item"),
                        "strict_correct": result.get("correct"),
                        "system_correct": result.get("system_correct"),
                        "explicit_nonclassifiable": result.get(
                            "explicit_nonclassifiable"
                        ),
                        "label_margin": result.get("label_margin"),
                        "label_softmax_top1": result.get("label_softmax_top1"),
                        "label_entropy": result.get("label_entropy"),
                        "top5": result.get("top5"),
                        "reasoning_n_tokens": result.get("reasoning_n_tokens"),
                        "reasoning_sha256_16": result.get("reasoning_sha256_16"),
                        "manual_expected_system": None,
                        "manual_expected_item": None,
                        "manual_correct_override": None,
                        "manual_note": None,
                    }
                )

        reasoning_wide: dict[str, dict] = {}
        for row in reasoning_rows:
            if row["role"] != args.wide_role:
                continue
            record = reasoning_wide.setdefault(
                row["record_id"],
                {
                    key: row[key]
                    for key in (
                        "record_id",
                        "source_id",
                        "source_row",
                        "text",
                        "n_words",
                        "annotation_source",
                        "gold_source_item",
                        "gold_grade",
                        "gold_choice_id",
                        "gold_system",
                        "gold_item",
                    )
                },
            )
            model_block = record.setdefault("models", {}).setdefault(row["model"], {})
            model_block[row["reasoning_mode"]] = {
                "model_choice_id": row["model_choice_id"],
                "model_choice_system": row["model_choice_system"],
                "model_choice_item": row["model_choice_item"],
                "strict_correct": row["strict_correct"],
                "system_correct": row["system_correct"],
                "explicit_nonclassifiable": row["explicit_nonclassifiable"],
                "label_margin": row["label_margin"],
                "label_softmax_top1": row["label_softmax_top1"],
                "label_entropy": row["label_entropy"],
                "reasoning_n_tokens": row["reasoning_n_tokens"],
                "reasoning_backend": row["reasoning_backend"],
                "native_thinking_enabled": row["native_thinking_enabled"],
            }

        reasoning_long_path = args.outdir / "reasoning_audit_long.jsonl"
        reasoning_wide_path = args.outdir / "reasoning_audit_wide.jsonl"
        with reasoning_long_path.open("w", encoding="utf-8") as handle:
            for row in reasoning_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with reasoning_wide_path.open("w", encoding="utf-8") as handle:
            for row in sorted(
                reasoning_wide.values(), key=lambda value: int(value["source_row"])
            ):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["reasoning"] = {
            "models": reasoning_models,
            "result_dataset_hash_matches": reasoning_hash_matches,
            "n_long_rows": len(reasoning_rows),
            "n_wide_rows": len(reasoning_wide),
            "wide_role": args.wide_role,
            "files": {
                "long": str(reasoning_long_path),
                "wide": str(reasoning_wide_path),
            },
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{len(reasoning_rows)} reasoning rows -> {reasoning_long_path}")
        print(f"{len(reasoning_wide)} reasoning wide rows -> {reasoning_wide_path}")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
