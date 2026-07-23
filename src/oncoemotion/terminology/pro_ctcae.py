"""PRO-CTCAE term library loader.

Loads ``terminology/pro_ctcae_terms.json`` (produced by
``scripts/build_terminology.py``) into validated, provenance-aware objects.

The library exposes *match entries* — normalized surface strings tagged with the
canonical id and provenance (official / reviewed / auto / synthetic / patient
phrase). Provenance is preserved so downstream code can decide which buckets to
trust and so evaluation can report which provenance produced a match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Provenance kinds, from most to least trusted.
PROVENANCE_ORDER = (
    "official_italian_label",
    "canonical_english",
    "reviewed_synonym",
    "auto_synonym",
    "synthetic_synonym",
    "synthetic_patient_phrase",
)


@dataclass(frozen=True)
class MatchEntry:
    """A single surface string that can match to a canonical term."""

    canonical_id: str
    surface: str
    provenance: str


@dataclass
class PROCTCAETerm:
    canonical_id: str
    canonical_english: str
    category: str
    attributes: list[str]
    official_italian_labels: list[str] = field(default_factory=list)
    reviewed_synonyms: list[str] = field(default_factory=list)
    auto_synonyms: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)  # synthetic dev synonyms
    common_patient_phrases: list[str] = field(default_factory=list)
    negation_examples: list[str] = field(default_factory=list)
    exclusion_examples: list[str] = field(default_factory=list)
    provenance: str = ""
    source_version: str = ""
    source_reference: str = ""

    def match_entries(self) -> list[MatchEntry]:
        """Surface strings for this term, each tagged with provenance."""
        out: list[MatchEntry] = []
        buckets = [
            ("official_italian_label", self.official_italian_labels),
            ("canonical_english", [self.canonical_english]),
            ("reviewed_synonym", self.reviewed_synonyms),
            ("auto_synonym", self.auto_synonyms),
            ("synthetic_synonym", self.synonyms),
            ("synthetic_patient_phrase", self.common_patient_phrases),
        ]
        for provenance, values in buckets:
            for v in values:
                if v and v.strip():
                    out.append(MatchEntry(self.canonical_id, v.strip(), provenance))
        return out


class PROCTCAELibrary:
    """In-memory PRO-CTCAE library with id/index accessors."""

    def __init__(self, terms: list[PROCTCAETerm], metadata: dict | None = None):
        self.terms = terms
        self.metadata = metadata or {}
        self._by_id = {t.canonical_id: t for t in terms}
        if len(self._by_id) != len(terms):
            raise ValueError("Duplicate canonical_id detected in PRO-CTCAE library")

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[PROCTCAETerm]:
        return iter(self.terms)

    def __contains__(self, canonical_id: str) -> bool:
        return canonical_id in self._by_id

    def get(self, canonical_id: str) -> PROCTCAETerm:
        return self._by_id[canonical_id]

    @property
    def canonical_ids(self) -> list[str]:
        return [t.canonical_id for t in self.terms]

    def all_match_entries(self, include_synthetic: bool = True) -> list[MatchEntry]:
        """All match entries across the library.

        ``include_synthetic=False`` drops synthetic buckets so that only
        official / reviewed data is used (useful once real data exists).
        """
        synthetic = {"synthetic_synonym", "synthetic_patient_phrase"}
        out: list[MatchEntry] = []
        for term in self.terms:
            for e in term.match_entries():
                if not include_synthetic and e.provenance in synthetic:
                    continue
                out.append(e)
        return out


def _default_path() -> Path:
    # <repo>/terminology/pro_ctcae_terms.json
    return Path(__file__).resolve().parents[3] / "terminology" / "pro_ctcae_terms.json"


def load_pro_ctcae(path: str | Path | None = None) -> PROCTCAELibrary:
    """Load the PRO-CTCAE library from JSON.

    Raises FileNotFoundError with actionable guidance if the file is missing —
    run ``python scripts/build_terminology.py`` to (re)generate it.
    """
    p = Path(path) if path is not None else _default_path()
    if not p.exists():
        raise FileNotFoundError(
            f"PRO-CTCAE terms file not found at {p}. "
            "Generate it with: python scripts/build_terminology.py"
        )
    doc = json.loads(p.read_text(encoding="utf-8"))
    raw_terms = doc.get("terms", doc if isinstance(doc, list) else [])
    terms = [
        PROCTCAETerm(
            canonical_id=r["canonical_id"],
            canonical_english=r["canonical_english"],
            category=r.get("category", ""),
            attributes=list(r.get("attributes", [])),
            official_italian_labels=list(r.get("official_italian_labels", [])),
            reviewed_synonyms=list(r.get("reviewed_synonyms", [])),
            auto_synonyms=list(r.get("auto_synonyms", [])),
            synonyms=list(r.get("synonyms", [])),
            common_patient_phrases=list(r.get("common_patient_phrases", [])),
            negation_examples=list(r.get("negation_examples", [])),
            exclusion_examples=list(r.get("exclusion_examples", [])),
            provenance=r.get("provenance", ""),
            source_version=r.get("source_version", ""),
            source_reference=r.get("source_reference", ""),
        )
        for r in raw_terms
    ]
    metadata = {k: v for k, v in doc.items() if k != "terms"} if isinstance(doc, dict) else {}
    return PROCTCAELibrary(terms, metadata)
