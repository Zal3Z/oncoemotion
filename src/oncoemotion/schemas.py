"""Pydantic I/O schemas for the mapper (spec section 4).

The output schema mirrors the specification exactly. An optional
``analysis_meta`` field carries non-clinical diagnostics (retrieval scores,
calibration flags, review reasons) for the dashboard and logging; it never
affects the clinical fields.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ------------------------------- enums ------------------------------------- #
Assertion = Literal["present", "negated", "uncertain", "hypothetical"]
Temporality = Literal["current", "past", "resolved", "unknown"]
Experiencer = Literal["patient", "other", "unknown"]

PROStatus = Literal[
    "EXACT_PRO_MATCH",
    "PRO_MATCH_WITH_AMBIGUITY",
    "MULTIPLE_POSSIBLE_PRO_MATCHES",
    "NO_DIRECT_PRO_MATCH",
    "NEGATED_SYMPTOM",
    "INSUFFICIENT_CONTEXT",
    "OUT_OF_SCOPE",
]
CTCAEStatus = Literal["MATCH", "AMBIGUOUS", "NO_MATCH", "NOT_EVALUATED"]


# ------------------------------- input ------------------------------------- #
class OptionalContext(BaseModel):
    time_window: Optional[str] = None
    treatment: Optional[str] = None
    patient_metadata: Optional[dict] = None


class MapRequest(BaseModel):
    record_id: str
    text: str
    language: str = "it"
    optional_context: OptionalContext = Field(default_factory=OptionalContext)


# ------------------------------- output ------------------------------------ #
class ClinicalMention(BaseModel):
    span: str
    start: int = 0
    end: int = 0
    assertion: Assertion = "present"
    temporality: Temporality = "current"
    experiencer: Experiencer = "patient"


class PROPrediction(BaseModel):
    canonical_id: str
    term: str
    probability: float = 0.0
    evidence_spans: list[str] = Field(default_factory=list)
    applicable_attributes: list[str] = Field(default_factory=list)
    explanation: str = ""


class PROResult(BaseModel):
    status: PROStatus
    predictions: list[PROPrediction] = Field(default_factory=list)


class CTCAEPrediction(BaseModel):
    ctcae_id: str
    term: str
    probability: float = 0.0
    evidence_spans: list[str] = Field(default_factory=list)
    explanation: str = ""


class CTCAEResult(BaseModel):
    version: str = "6.0"
    status: CTCAEStatus = "NOT_EVALUATED"
    predictions: list[CTCAEPrediction] = Field(default_factory=list)
    grade: Optional[int] = None


class SafetyResult(BaseModel):
    urgent_human_review: bool = False
    reason: Optional[str] = None


class MapResponse(BaseModel):
    record_id: str
    normalized_text: str
    clinical_mentions: list[ClinicalMention] = Field(default_factory=list)
    pro_ctcae: PROResult
    ctcae: CTCAEResult = Field(default_factory=CTCAEResult)
    safety: SafetyResult = Field(default_factory=SafetyResult)
    abstain: bool = False
    abstention_reason: Optional[str] = None
    # Non-clinical diagnostics (scores, calibration flags, review reasons):
    analysis_meta: Optional[dict] = None
