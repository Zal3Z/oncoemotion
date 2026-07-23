"""Shared retrieval datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndexEntry:
    """A searchable surface string tied to a target id."""

    target_id: str          # PRO canonical_id or CTCAE id
    term: str               # display term for the target
    surface: str            # the surface string (display form)
    surface_match: str      # normalized match form
    provenance: str = ""


@dataclass
class Candidate:
    target_id: str
    term: str
    score: float
    matched_surface: str = ""
    provenance: str = ""
    evidence_start: int = 0
    evidence_end: int = 0
    signals: dict = field(default_factory=dict)

    @property
    def evidence_span(self) -> tuple[int, int]:
        return (self.evidence_start, self.evidence_end)
