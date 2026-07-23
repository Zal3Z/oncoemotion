#!/usr/bin/env python
"""Build ``terminology/pro_ctcae_terms.json`` from the canonical source of truth.

This is deterministic and reproducible: it encodes the 80 canonical PRO-CTCAE
symptom terms (English identifiers + attributes, from spec section 3) and emits
the per-term JSON records in the schema required by the specification.

Italian synonyms / patient phrases are SYNTHETIC developer placeholders (clearly
labelled) and only present for a subset of terms — enough to make the baseline
mapper and the mandatory regression tests meaningful. ``official_italian_labels``
is left empty until the official PRO-CTCAE Italian PDF is loaded.

Usage:
    python scripts/build_terminology.py [--out terminology/pro_ctcae_terms.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oncoemotion.terminology.canonical import (  # noqa: E402
    CANONICAL_TERMS,
    EXPECTED_TERM_COUNT,
    SYNTHETIC_SEEDS,
    attribute_names,
    canonical_id,
)

SCHEMA_VERSION = "pro_ctcae_terms/1.1"
DEFAULT_OFFICIAL = (
    Path(__file__).resolve().parents[1] / "terminology" / "official" / "pro_ctcae_italian_labels.json"
)


def load_official_labels(path: Path) -> dict:
    """Load extracted official Italian labels, if present (else empty)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_records(official: dict | None = None) -> list[dict]:
    if len(CANONICAL_TERMS) != EXPECTED_TERM_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_TERM_COUNT} canonical terms, found {len(CANONICAL_TERMS)}"
        )
    official_terms = (official or {}).get("terms", {})
    records: list[dict] = []
    seen_ids: set[str] = set()
    for ordinal, english, category, letters in CANONICAL_TERMS:
        cid = canonical_id(ordinal)
        if cid in seen_ids:
            raise AssertionError(f"Duplicate canonical_id: {cid}")
        seen_ids.add(cid)
        seed = SYNTHETIC_SEEDS.get(cid, {})
        off = official_terms.get(cid, {})
        official_label = off.get("official_italian_label")
        official_labels = [official_label] if official_label else []
        auto_syn = list(off.get("derived_synonyms", []))
        provenance = []
        if off:
            provenance.append("official_it_pdf")
        if seed:
            provenance.append("synthetic_dev")
        records.append(
            {
                "canonical_id": cid,
                "canonical_english": english,
                # Official Italian labels from the NCI PRO-CTCAE Italian PDF
                # (empty only if the official file was not present at build time).
                "official_italian_labels": official_labels,
                "category": category,
                "attributes": attribute_names(letters),
                # --- provenance-separated synonym buckets (spec section 3) ---
                "reviewed_synonyms": [],                       # await human review
                "auto_synonyms": auto_syn,                     # derived from official label
                "synonyms": list(seed.get("synonyms", [])),    # synthetic dev
                "common_patient_phrases": list(seed.get("common_patient_phrases", [])),
                "negation_examples": list(seed.get("negation_examples", [])),
                "exclusion_examples": list(seed.get("exclusion_examples", [])),
                "provenance": "+".join(provenance) if provenance else "canonical_only",
                "source_version": (official or {}).get("source_version", ""),
                "source_reference": (official or {}).get("source", ""),
            }
        )
    return records


def build_document(official_path: Path | None = None) -> dict:
    official = load_official_labels(official_path or DEFAULT_OFFICIAL)
    has_official = bool(official)
    return {
        "schema_version": SCHEMA_VERSION,
        "term_count": EXPECTED_TERM_COUNT,
        "provenance_note": (
            "canonical_english + attributes are the official PRO-CTCAE identifiers "
            "(spec section 3). official_italian_labels + auto_synonyms are "
            + ("from the NCI PRO-CTCAE Italian Item Library (official). " if has_official
               else "EMPTY (official Italian PDF not found at build time). ")
            + "synonyms / common_patient_phrases / negation_examples / "
            "exclusion_examples are SYNTHETIC developer placeholder data. "
            "reviewed_synonyms await clinical review. NCI PRO-CTCAE Terms of Use "
            "apply to the Italian labels."
        ),
        "source_version": official.get("source_version", ""),
        "source_reference": official.get("source", ""),
        "official_labels_loaded": has_official,
        "terms": build_records(official),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parents[1] / "terminology" / "pro_ctcae_terms.json"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    doc = build_document()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(doc['terms'])} PRO-CTCAE terms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
