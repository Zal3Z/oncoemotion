"""CTCAE dictionary loader (separate from PRO-CTCAE).

The CTCAE dictionary is consulted ONLY as an explicit fallback when there is no
direct PRO-CTCAE match (spec section 2). Terms are never invented: the loader
reads an official file if provided, otherwise an explicitly-labelled synthetic
placeholder may be requested for development and tests.

Grading is intentionally *not* performed here. Grading is a separate module
(spec section 2 D) that must abstain when required information is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class CTCAETerm:
    ctcae_id: str
    term: str
    soc: str = ""
    italian_labels: list[str] = field(default_factory=list)
    grade_definitions: dict[str, str] = field(default_factory=dict)
    definition: str = ""
    grade_block_raw: str = ""
    is_synthetic: bool = False

    def match_entries(self) -> list[tuple[str, str]]:
        """Return (surface, kind) pairs; kind marks provenance."""
        if self.is_synthetic:
            it_kind = "synthetic_label"
        else:
            it_kind = "italian_bridge"
        out: list[tuple[str, str]] = [(self.term, "english_term")]
        out.extend((lbl, it_kind) for lbl in self.italian_labels if lbl.strip())
        return out


class CTCAEDictionary:
    def __init__(self, terms: list[CTCAETerm], version: str, is_synthetic: bool, metadata: dict | None = None):
        self.terms = terms
        self.version = version
        self.is_synthetic = is_synthetic
        self.metadata = metadata or {}
        self._by_id = {t.ctcae_id: t for t in terms}

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[CTCAETerm]:
        return iter(self.terms)

    def get(self, ctcae_id: str) -> CTCAETerm:
        return self._by_id[ctcae_id]


def _terminology_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "terminology"


def _synthetic_path() -> Path:
    return _terminology_dir() / "ctcae_v6_synthetic.json"


def _default_bridge_path() -> Path:
    return _terminology_dir() / "ctcae_italian_bridge.json"


def _apply_italian_bridge(terms: list[CTCAETerm], bridge_path: Path) -> int:
    """Attach Italian surface strings to matching CTCAE terms. Skip unknown
    targets (never invents terms). Returns the number of targets applied."""
    if not bridge_path.exists():
        return 0
    bridge = json.loads(bridge_path.read_text(encoding="utf-8")).get("map", {})
    by_term = {t.term.lower(): t for t in terms}
    applied = 0
    for english, surfaces in bridge.items():
        t = by_term.get(english.lower())
        if t is None:
            continue  # target not in dictionary -> skip, do not invent
        t.italian_labels = list(dict.fromkeys([*t.italian_labels, *surfaces]))
        applied += 1
    return applied


def load_ctcae(
    path: str | Path | None = None,
    allow_synthetic: bool = False,
    italian_bridge: str | Path | None = None,
) -> CTCAEDictionary:
    """Load a CTCAE dictionary.

    * ``path`` given -> load that official file (JSON in the documented schema).
    * ``path is None`` and ``allow_synthetic`` -> load the labelled synthetic
      placeholder (development / tests only).
    * ``path is None`` and not ``allow_synthetic`` -> FileNotFoundError.

    The official CTCAE v6.0 is English; an Italian bridge (default
    ``terminology/ctcae_italian_bridge.json``) supplies Italian surface strings
    for the fallback. Unknown bridge targets are skipped.
    """
    if path is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "No CTCAE file provided. Supply the official CTCAE v6.0 file path, "
                "or pass allow_synthetic=True to use the clearly-labelled synthetic "
                "placeholder (development/tests only)."
            )
        p = _synthetic_path()
    else:
        p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CTCAE file not found at {p}")

    doc = json.loads(p.read_text(encoding="utf-8"))
    version = str(doc.get("version", "unknown"))
    is_synthetic = "SYNTHETIC" in version.upper() or bool(doc.get("synthetic"))
    terms = [
        CTCAETerm(
            ctcae_id=r["ctcae_id"],
            term=r["term"],
            soc=r.get("soc", ""),
            italian_labels=list(r.get("italian_synthetic_labels", r.get("italian_labels", []))),
            grade_definitions=dict(r.get("grade_definitions", {})),
            definition=r.get("definition", ""),
            grade_block_raw=r.get("grade_block_raw", ""),
            is_synthetic=is_synthetic,
        )
        for r in doc.get("terms", [])
    ]
    # Attach the Italian bridge (for both official and synthetic dicts).
    bridge_path = Path(italian_bridge) if italian_bridge is not None else _default_bridge_path()
    n_bridge = _apply_italian_bridge(terms, bridge_path)
    metadata = {k: v for k, v in doc.items() if k != "terms"}
    metadata["italian_bridge_targets_applied"] = n_bridge
    return CTCAEDictionary(terms, version=version, is_synthetic=is_synthetic, metadata=metadata)
