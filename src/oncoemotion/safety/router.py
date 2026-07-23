"""Independent safety router.

Runs *separately* from the mapping pipeline (spec section 5, step 9). It flags
inputs indicating potential immediate risk (self-harm / suicidal ideation) for an
organization-defined human workflow. It never generates clinical emergency
instructions and never blocks the (separate) terminology mapping — e.g. for
"suicidio" the mapper still returns NO_DIRECT_PRO_MATCH while safety raises the
urgent flag.

Detection is metalanguage-aware: obvious quoting / classification framing
("la parola da classificare è ...") is treated cautiously. This is a
conservative keyword+context baseline, not a validated risk model.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


def _fold(text: str) -> str:
    d = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in d if unicodedata.category(ch) != "Mn")


# Self-harm / suicidal-ideation cues (Italian + a few English).
SELF_HARM_CUES = [_fold(x) for x in (
    "suicidio", "suicidarmi", "mi suicido", "voglio morire", "farla finita",
    "uccidermi", "ammazzarmi", "togliermi la vita", "non voglio piu vivere",
    "farmi del male", "autolesionismo", "tagliarmi", "la mia vita non ha senso",
    "meglio morire", "kill myself", "suicide", "self-harm", "end my life",
)]

# Framing that suggests the term is being *mentioned*, not asserted.
METALANGUAGE_CUES = [_fold(x) for x in (
    "la parola da classificare", "la frase da classificare", "classifica la parola",
    "il termine", "esempio:", "definizione di", "significa", "come si scrive",
)]


@dataclass
class SafetyDecision:
    urgent_human_review: bool = False
    reason: str | None = None
    matched_cues: list[str] = field(default_factory=list)
    metalanguage_context: bool = False


class SafetyRouter:
    def __init__(self, extra_cues: list[str] | None = None):
        self.cues = list(SELF_HARM_CUES)
        if extra_cues:
            self.cues.extend(_fold(c) for c in extra_cues)

    def check(self, text: str) -> SafetyDecision:
        folded = _fold(text)
        matched = [c for c in self.cues if c in folded]
        meta = any(m in folded for m in METALANGUAGE_CUES)
        if not matched:
            return SafetyDecision(False, None, [], meta)
        # Even under metalanguage framing we err toward review, but record context.
        reason = (
            "Potential self-harm / suicidal-ideation content detected; routed to "
            "organization-defined human review workflow."
        )
        if meta:
            reason += " (metalanguage/classification framing also detected — verify.)"
        return SafetyDecision(True, reason, matched, meta)
