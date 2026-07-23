"""Deterministic lexical + fuzzy retriever.

Combines exact match, whole-word substring containment, and a fuzzy token
similarity. Uses :mod:`rapidfuzz` when installed, otherwise falls back to
:class:`difflib.SequenceMatcher` so the baseline runs with zero ML dependencies.
Results are identical for a given input regardless of batch size.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from oncoemotion.preprocessing.normalize import Normalizer
from oncoemotion.retrieval.base import Candidate, IndexEntry

try:  # optional, faster + better fuzzy
    from rapidfuzz import fuzz as _rf_fuzz

    _HAVE_RAPIDFUZZ = True
except Exception:  # pragma: no cover - exercised only when rapidfuzz absent
    _rf_fuzz = None
    _HAVE_RAPIDFUZZ = False


def fuzzy_ratio(a: str, b: str) -> float:
    """Token-aware fuzzy similarity in [0, 1].

    Uses ``token_set_ratio`` (rapidfuzz) rather than ``partial_ratio``:
    partial_ratio is far too permissive on short strings (e.g. it scores
    "ansia" vs "emicrania" at 0.89), which would create spurious symptom
    matches. token_set_ratio keeps such unrelated short tokens well separated
    (~0.57) while still rewarding true near-synonyms (~0.87+).
    """
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


class LexicalFuzzyRetriever:
    """Retrieve candidate targets for a text segment."""

    def __init__(self, entries: list[IndexEntry], normalizer: Normalizer | None = None):
        self.entries = entries
        self.normalizer = normalizer or Normalizer()

    def _word_span(self, needle_match: str, haystack_display: str) -> tuple[int, int] | None:
        """Locate ``needle`` in the display haystack (accent/case-insensitive)."""
        hay_match = self.normalizer.to_match(haystack_display)
        pattern = re.compile(r"\b" + re.escape(needle_match) + r"\b")
        m = pattern.search(hay_match)
        if not m:
            idx = hay_match.find(needle_match)
            if idx < 0:
                return None
            return (idx, idx + len(needle_match))
        return (m.start(), m.end())

    def score_entry(self, seg_match: str, entry: IndexEntry) -> tuple[float, str]:
        """Return (score, signal) for one index entry against a segment."""
        surf = entry.surface_match
        if not surf:
            return 0.0, "empty"
        if seg_match == surf:
            return 1.0, "exact"
        # whole-word substring containment
        if re.search(r"\b" + re.escape(surf) + r"\b", seg_match):
            coverage = len(surf) / max(len(seg_match), 1)
            return min(0.90 + 0.10 * coverage, 0.99), "substring"
        return fuzzy_ratio(seg_match, surf), "fuzzy"

    def retrieve(
        self,
        seg_display: str,
        seg_offset: int = 0,
        top_k: int = 5,
        min_score: float = 0.30,
        fuzzy_floor: float = 0.0,
    ) -> list[Candidate]:
        """Return top-k candidates (max score aggregated per target id).

        ``fuzzy_floor`` gates candidates whose only evidence is fuzzy similarity
        (exact / whole-word-substring matches are exempt), suppressing spurious
        near-string matches that must not be force-coded.
        """
        seg_match = self.normalizer.to_match(seg_display)
        best: dict[str, Candidate] = {}
        for entry in self.entries:
            score, signal = self.score_entry(seg_match, entry)
            if score < min_score:
                continue
            if signal == "fuzzy" and score < fuzzy_floor:
                continue
            cur = best.get(entry.target_id)
            if cur is None or score > cur.score:
                span = self._word_span(entry.surface_match, seg_display)
                if span is None:
                    ev_start, ev_end = seg_offset, seg_offset + len(seg_display)
                else:
                    ev_start, ev_end = seg_offset + span[0], seg_offset + span[1]
                best[entry.target_id] = Candidate(
                    target_id=entry.target_id,
                    term=entry.term,
                    score=score,
                    matched_surface=entry.surface,
                    provenance=entry.provenance,
                    evidence_start=ev_start,
                    evidence_end=ev_end,
                    signals={"signal": signal},
                )
        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
