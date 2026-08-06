"""Joint PRO-CTCAE / CTCAE / non-classifiable decision utilities.

The original real-field endpoint intentionally treated CTCAE-only rows as having
no direct PRO target.  This module adds a separate behavioural extension: the
model may choose a PRO-CTCAE item, a CTCAE item observed in the validated source,
or an explicit ``NON_CLASSIFICABILE`` option.  Keeping the extension separate
prevents the original PRO-only endpoint from being redefined after results exist.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping

from oncoemotion.clinical.classify import Candidate, build_candidates

PRO_PREFIX = "PRO::"
CTCAE_PREFIX = "CTCAE::"
NON_CLASSIFICABILE_ID = "NON_CLASSIFICABILE"


@dataclass(frozen=True)
class JointChoice:
    choice_id: str
    system: str
    item: str | None
    pro_id: str | None = None


def ctcae_choice_id(term: str) -> str:
    """Stable identifier that retains the validated CTCAE display term."""
    return f"{CTCAE_PREFIX}{term.strip()}"


def pro_choice_id(pro_id: str) -> str:
    return f"{PRO_PREFIX}{pro_id.strip()}"


def gold_joint_choice(record: Mapping[str, object]) -> JointChoice:
    source = str(record.get("annotation_source") or "").strip()
    if source == "PRO-CTCAE" and record.get("gold_pro_id"):
        pid = str(record["gold_pro_id"])
        return JointChoice(pro_choice_id(pid), "PRO-CTCAE", str(record.get("gold_pro_term") or pid), pid)
    if source == "CTCAE v5" and record.get("gold_ctcae_term"):
        term = str(record["gold_ctcae_term"]).strip()
        return JointChoice(ctcae_choice_id(term), "CTCAE", term, None)
    return JointChoice(NON_CLASSIFICABILE_ID, "NON_CLASSIFICABILE", None, None)


def decode_joint_choice(choice_id: str, display_term: str | None = None) -> JointChoice:
    if choice_id == NON_CLASSIFICABILE_ID:
        return JointChoice(choice_id, "NON_CLASSIFICABILE", None, None)
    if choice_id.startswith(PRO_PREFIX):
        pid = choice_id[len(PRO_PREFIX):]
        return JointChoice(choice_id, "PRO-CTCAE", display_term or pid, pid)
    if choice_id.startswith(CTCAE_PREFIX):
        term = choice_id[len(CTCAE_PREFIX):]
        return JointChoice(choice_id, "CTCAE", display_term or term, None)
    raise ValueError(f"unknown joint choice id: {choice_id!r}")


def build_joint_candidates(adapter, pro_library, ctcae_terms: Iterable[str]) -> list[Candidate]:
    """Build auditable constrained choices for the mixed validated label space.

    The ontology prefix is part of every scored continuation.  Without it, an
    overlapping term such as pain could not distinguish a PRO choice from a CTCAE
    choice.  PRO surfaces reuse the existing terminology builder; CTCAE choices are
    limited to the exact item labels already validated in the private source.
    """
    tok = adapter.tokenizer
    choices: list[Candidate] = []

    # Keep the compact terminology surfaces.  The joint task is already larger
    # than the 80-way PRO task, so one patient phrase per PRO item is enough.
    for candidate in build_candidates(adapter, pro_library, forms=("it", "en", "phrase")):
        surfaces = [f"PRO-CTCAE | {surface}" for surface in candidate.surfaces]
        surface_ids = [tok(surface, add_special_tokens=False).input_ids for surface in surfaces]
        choices.append(
            Candidate(
                canonical_id=pro_choice_id(candidate.canonical_id),
                term_en=candidate.term_en,
                surfaces=surfaces,
                surface_ids=surface_ids,
            )
        )

    for term in sorted({str(value).strip() for value in ctcae_terms if str(value).strip()}):
        surfaces = [f"CTCAE | {term}", f"CTCAE | {term.lower()}"]
        surfaces = list(dict.fromkeys(surfaces))
        choices.append(
            Candidate(
                canonical_id=ctcae_choice_id(term),
                term_en=term,
                surfaces=surfaces,
                surface_ids=[tok(surface, add_special_tokens=False).input_ids for surface in surfaces],
            )
        )

    abstain_surfaces = [
        "NON_CLASSIFICABILE",
        "NON CLASSIFICABILE",
        "NON SO",
    ]
    choices.append(
        Candidate(
            canonical_id=NON_CLASSIFICABILE_ID,
            term_en="Non classificabile",
            surfaces=abstain_surfaces,
            surface_ids=[tok(surface, add_special_tokens=False).input_ids for surface in abstain_surfaces],
        )
    )
    return choices


def redacted_reasoning_digest(text: str) -> str | None:
    """Non-reversible audit handle for a generated deliberation."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]

