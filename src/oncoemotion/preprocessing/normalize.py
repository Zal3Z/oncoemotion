"""Unicode normalization and light spelling normalization for Italian input.

Two forms are produced:
  * display form  — NFC, whitespace-collapsed, trimmed (goes to normalized_text)
  * match form    — lowercased, accent-folded, light typo fixes (matching only)

Character offsets returned by :meth:`Normalizer.to_display` are 1:1 with the
input after whitespace collapsing so that spans remain interpretable.
"""

from __future__ import annotations

import re
import unicodedata

# Minimal, conservative typo map (extend with data-driven rules later).
_TYPO_MAP = {
    "nasuea": "nausea",
    "nausa": "nausea",
    "vomitp": "vomito",
    "diarea": "diarrea",
    "mal di testà": "mal di testa",
}

_WS = re.compile(r"\s+")


class Normalizer:
    def __init__(self, fold_accents: bool = True, typo_map: dict[str, str] | None = None):
        self.fold_accents = fold_accents
        self.typo_map = typo_map if typo_map is not None else dict(_TYPO_MAP)

    @staticmethod
    def _nfc(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def to_display(self, text: str) -> str:
        """NFC + collapse internal whitespace + strip. Preserves case & accents."""
        return _WS.sub(" ", self._nfc(text)).strip()

    def _fold(self, text: str) -> str:
        # Decompose and drop combining marks (accent folding).
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

    def to_match(self, text: str) -> str:
        """Lowercase, optional accent fold, whitespace collapse, typo fixes."""
        t = self.to_display(text).lower()
        if self.fold_accents:
            t = self._fold(t)
        for wrong, right in self.typo_map.items():
            key = self._fold(wrong) if self.fold_accents else wrong
            t = t.replace(key, self._fold(right) if self.fold_accents else right)
        return t
