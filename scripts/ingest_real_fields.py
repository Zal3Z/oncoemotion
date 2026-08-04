#!/usr/bin/env python
"""Ingest clinician-validated real free-text fields without altering the workbook.

The source spreadsheet carries one row per clinical association. Identical text is
not assumed to mean a duplicate assessment: the current source contains identical
strings with different item/value associations. Each Excel row therefore receives
its own ``record_id``; ``source_id`` is used only as a text-cluster key for robust
inference and efficient model execution.

PRO label names are resolved through an explicit, versioned crosswalk rather than a
dictionary hidden in this script. The conversion fails on an unmapped PRO label so
that a gold association can never disappear silently.

Usage:
    python scripts/ingest_real_fields.py --xlsx sinomi_campi_aperti.xlsx
    python scripts/ingest_real_fields.py --xlsx ... --check-only
    python scripts/ingest_real_fields.py --xlsx ... --min-words 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]

EXACT = "EXACT_PRO_MATCH"
NOD = "NO_DIRECT_PRO_MATCH"
INS = "INSUFFICIENT_CONTEXT"
REQUIRED_COLUMNS = {
    "campo_aperto",
    "valore_associato",
    "fonte_associazione",
    "item_associato",
}


def _norm_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _normalise_text_for_id(text: str) -> str:
    # Preserve case, punctuation and internal whitespace because they can change
    # tokenization and therefore model output. Only Unicode form and edge whitespace
    # are normalized.
    return unicodedata.normalize("NFC", text).strip()


def _source_id(text: str) -> str:
    """Stable, non-reversible cluster key for the normalized patient string."""
    return hashlib.sha256(_normalise_text_for_id(text).encode("utf-8")).hexdigest()[:16]


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_crosswalk(path: Path, terms_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "real_field_association_map/1.0":
        raise ValueError(f"unsupported crosswalk schema: {raw.get('schema_version')!r}")
    mappings = raw.get("mappings") or {}
    if not mappings:
        raise ValueError("crosswalk has no mappings")

    terms = json.loads(terms_path.read_text(encoding="utf-8"))
    by_id = {term["canonical_id"]: term for term in terms["terms"]}
    invalid = sorted(
        (label, entry.get("canonical_id"))
        for label, entry in mappings.items()
        if entry.get("canonical_id") not in by_id
    )
    if invalid:
        raise ValueError(f"crosswalk contains unknown PRO identifiers: {invalid[:5]}")
    return mappings, by_id


def convert_rows(
    rows: Iterable[Mapping[str, object]],
    mappings: Mapping[str, Mapping[str, object]],
    terms_by_id: Mapping[str, Mapping[str, object]],
    *,
    min_words: int = 0,
) -> list[dict]:
    """Convert spreadsheet-shaped mappings into the model dataset schema."""
    records: list[dict] = []
    unmapped: Counter[str] = Counter()
    for position, row in enumerate(rows, start=2):
        text = row.get("campo_aperto")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        n_words = len(text.split())
        if min_words and n_words < min_words:
            continue

        source = str(row.get("fonte_associazione") or "").strip()
        item_value = row.get("item_associato")
        source_item = None if _missing(item_value) else str(item_value).strip()
        key = _norm_label(source_item or "")
        pro_id = None
        mapping_note = None
        if source == "PRO-CTCAE":
            entry = mappings.get(key)
            if entry is None:
                unmapped[source_item or "<missing>"] += 1
                continue
            pro_id = str(entry["canonical_id"])
            mapping_note = entry.get("mapping_note")
            gold_class, category, status = "term", EXACT, EXACT
        elif source == "CTCAE v5":
            gold_class, category, status = "abstain", NOD, NOD
        else:
            gold_class, category, status = "abstain", INS, INS

        grade_value = row.get("valore_associato")
        grade = None if _missing(grade_value) else int(grade_value)
        source_row = int(row.get("_source_row") or position)
        record_id = f"real_{source_row:05d}"
        sid = _source_id(text)
        records.append({
            "record_id": record_id,
            "pair_id": record_id,
            "framing": "real",
            "text": text,
            "language": "it",
            "category": category,
            "gold_class": gold_class,
            "gold_pro_id": pro_id,
            "gold_pro_term": terms_by_id[pro_id]["canonical_english"] if pro_id else None,
            "gold_pro_status": status,
            "gold_ctcae_term": source_item if source == "CTCAE v5" else None,
            "urgent": False,
            "grade": grade,
            "source_id": sid,
            "assessment_id": None,
            "source_row": source_row,
            "source_item": source_item,
            "n_words": n_words,
            "annotation_source": source,
            "crosswalk_note": mapping_note,
        })

    if unmapped:
        detail = ", ".join(f"{label} ({count})" for label, count in unmapped.most_common())
        raise ValueError(f"unmapped PRO-CTCAE source labels; update the crosswalk: {detail}")
    return records


def annotation_signature(record: Mapping[str, object]) -> tuple:
    return (
        record.get("annotation_source"),
        record.get("gold_class"),
        record.get("gold_pro_id"),
        record.get("gold_ctcae_term"),
        record.get("grade"),
    )


def duplicate_audit(records: Iterable[Mapping[str, object]]) -> dict:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record["source_id"])].append(record)
    duplicate_groups = {key: value for key, value in groups.items() if len(value) > 1}
    conflicting = {
        key: value for key, value in duplicate_groups.items()
        if len({annotation_signature(record) for record in value}) > 1
    }
    return {
        "unique_source_ids": len(groups),
        "duplicate_source_groups": len(duplicate_groups),
        "conflicting_annotation_groups": len(conflicting),
        "rows_in_duplicate_groups": sum(len(value) for value in duplicate_groups.values()),
        "rows_in_conflicting_groups": sum(len(value) for value in conflicting.values()),
    }


def deduplicate_identical_annotations(records: list[dict]) -> list[dict]:
    """Collapse only exact text+annotation repeats; reject conflicting text clusters."""
    audit = duplicate_audit(records)
    if audit["conflicting_annotation_groups"]:
        raise ValueError(
            "cannot --dedup: identical texts have conflicting source/item/value "
            f"associations in {audit['conflicting_annotation_groups']} groups"
        )
    seen: set[tuple] = set()
    kept = []
    for record in records:
        key = (record["source_id"], annotation_signature(record))
        if key not in seen:
            kept.append(record)
            seen.add(key)
    return kept


def summarize_records(records: list[dict]) -> dict:
    audit = duplicate_audit(records)
    categories = Counter(record["category"] for record in records)
    grades = Counter(record["grade"] for record in records if record["grade"] is not None)
    term_ids = {record["gold_pro_id"] for record in records if record["gold_pro_id"]}
    return {
        "records": len(records),
        "term_records": sum(record["gold_class"] == "term" for record in records),
        "abstain_records": sum(record["gold_class"] == "abstain" for record in records),
        "categories": dict(sorted(categories.items())),
        "grades": {str(key): value for key, value in sorted(grades.items())},
        "distinct_pro_ids": len(term_ids),
        "median_words": int(sorted(record["n_words"] for record in records)[len(records) // 2]),
        "records_ge_7_words": sum(record["n_words"] >= 7 for record in records),
        **audit,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_ROOT / "data/real/clinical_real.jsonl")
    ap.add_argument(
        "--crosswalk",
        type=Path,
        default=_ROOT / "terminology/real_field_association_map.json",
    )
    ap.add_argument(
        "--terms",
        type=Path,
        default=_ROOT / "terminology/pro_ctcae_terms.json",
    )
    ap.add_argument("--min-words", type=int, default=0)
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="collapse exact text+annotation repeats; fails if a text has conflicting annotations",
    )
    ap.add_argument("--expected-records", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    frame = pd.read_excel(args.xlsx)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        ap.error(f"workbook is missing required columns: {missing_columns}")
    rows = frame.to_dict(orient="records")
    for excel_row, row in enumerate(rows, start=2):
        row["_source_row"] = excel_row

    mappings, terms_by_id = load_crosswalk(args.crosswalk, args.terms)
    records = convert_rows(rows, mappings, terms_by_id, min_words=args.min_words)
    pre_dedup_audit = duplicate_audit(records)
    if args.dedup:
        records = deduplicate_identical_annotations(records)
    if args.expected_records and len(records) != args.expected_records:
        raise ValueError(
            f"record count mismatch: expected {args.expected_records}, observed {len(records)}"
        )

    summary = summarize_records(records)
    summary["pre_dedup_audit"] = pre_dedup_audit
    summary["source_sha256"] = _sha256(args.xlsx)
    summary["crosswalk_sha256"] = _sha256(args.crosswalk)
    summary["crosswalk_schema"] = "real_field_association_map/1.0"
    summary["min_words"] = args.min_words
    summary["deduplicated"] = bool(args.dedup)

    if not args.check_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary["dataset_sha256"] = _sha256(args.out)
        manifest = args.out.with_name(args.out.stem + "_manifest.json")
        manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(records)} records -> {args.out}")
        print(f"manifest -> {manifest}")
    else:
        print(f"check only: {len(records)} records")

    print(json.dumps({
        key: summary[key] for key in (
            "term_records",
            "abstain_records",
            "categories",
            "grades",
            "distinct_pro_ids",
            "unique_source_ids",
            "duplicate_source_groups",
            "conflicting_annotation_groups",
            "records_ge_7_words",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
