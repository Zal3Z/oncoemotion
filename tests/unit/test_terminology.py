"""Terminology invariants (spec section 16)."""

from __future__ import annotations

from oncoemotion.terminology.canonical import (
    CANONICAL_TERMS,
    ATTRIBUTE_NAMES,
    attribute_names,
    canonical_id,
)

ALLOWED_ATTRS = set(ATTRIBUTE_NAMES.values())


def test_exactly_80_terms(pro_library):
    assert len(pro_library) == 80


def test_unique_canonical_ids(pro_library):
    ids = pro_library.canonical_ids
    assert len(ids) == len(set(ids)) == 80


def test_canonical_id_format(pro_library):
    for t in pro_library:
        assert t.canonical_id.startswith("PRO_")
        assert len(t.canonical_id) == 7  # PRO_ + 3 digits


def test_no_invented_terms(pro_library):
    """Every loaded English term is one of the canonical spec terms; nothing extra."""
    expected = {english for _, english, _, _ in CANONICAL_TERMS}
    loaded = {t.canonical_english for t in pro_library}
    assert loaded == expected


def test_attribute_association(pro_library):
    # Spot-check attribute mappings from the specification.
    cases = {
        "PRO_009": ["frequency", "severity"],       # Nausea [F,S]
        "PRO_033": ["presence"],                     # Nail discoloration [P]
        "PRO_017": ["frequency", "severity", "interference"],  # Abdominal pain
        "PRO_027": ["amount"],                       # Hair loss [A]
        "PRO_054": ["frequency", "severity", "interference"],  # Anxious
    }
    for cid, attrs in cases.items():
        assert pro_library.get(cid).attributes == attrs


def test_all_attributes_in_allowed_set(pro_library):
    for t in pro_library:
        assert set(t.attributes) <= ALLOWED_ATTRS
        assert t.attributes, f"{t.canonical_id} has no attributes"


def test_specific_ids(pro_library):
    assert pro_library.get("PRO_033").canonical_english == "Nail discoloration"
    assert pro_library.get("PRO_009").canonical_english == "Nausea"
    assert pro_library.get("PRO_054").canonical_english == "Anxious"
    assert canonical_id(33) == "PRO_033"
    assert attribute_names("FS") == ["frequency", "severity"]


def test_official_italian_labels_state(pro_library):
    """If the official IT PDF was loaded at build time, every term has a label;
    otherwise labels stay empty (never invented)."""
    loaded = pro_library.metadata.get("official_labels_loaded", False)
    if loaded:
        for t in pro_library:
            assert t.official_italian_labels, f"{t.canonical_id} missing official label"
        assert "NAUSEA" in pro_library.get("PRO_009").official_italian_labels[0].upper()
        assert pro_library.get("PRO_054").official_italian_labels == ["ANSIA"]
    else:
        for t in pro_library:
            assert t.official_italian_labels == []
