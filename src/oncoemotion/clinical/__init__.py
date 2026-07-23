"""Phase 3 — clinical measurement of emotion-like signals (spec sections 9-11).

Builds the PRO-CTCAE decision prompt with the teacher-forced output prefix,
captures activations at the measurement points (primary: point E, pre-decision),
projects them onto the Phase-2 emotion/control vectors, and analyses severity
gradients, confounder distinguishability, and persistence.
"""

from oncoemotion.clinical.prompt import (
    build_decision_prompt,
    TEACHER_PREFIX,
    NEUTRAL_FILLER,
)

__all__ = ["build_decision_prompt", "TEACHER_PREFIX", "NEUTRAL_FILLER"]
