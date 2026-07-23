"""Baseline PRO-CTCAE / CTCAE mapper (deterministic, no LLM required).

Implements the modular pipeline of spec section 5:
  1 normalization  2 segmentation  3 assertion/temporality/experiencer
  4 candidate generation (lexical + fuzzy)  5 (optional) reranking
  6 thresholded decision  7 mandatory abstention  8 separate CTCAE fallback
  9 independent safety routing  10 reproducible diagnostics

Design rule (spec section 5): the emotion-probing module must never alter this
baseline mapping. This module has no dependency on the interpretability stack.
"""

from __future__ import annotations

import unicodedata

from oncoemotion.config import MappingConfig
from oncoemotion.mapping.calibration import Calibrator, HeuristicCalibrator
from oncoemotion.preprocessing.assertion import detect_assertion_temporality
from oncoemotion.preprocessing.normalize import Normalizer
from oncoemotion.preprocessing.segment import segment_text
from oncoemotion.retrieval.base import Candidate, IndexEntry
from oncoemotion.retrieval.lexical_fuzzy import LexicalFuzzyRetriever, fuzzy_ratio
from oncoemotion.safety.router import SafetyRouter
from oncoemotion.schemas import (
    ClinicalMention,
    CTCAEPrediction,
    CTCAEResult,
    MapRequest,
    MapResponse,
    PROPrediction,
    PROResult,
    SafetyResult,
)
from oncoemotion.terminology.ctcae import CTCAEDictionary
from oncoemotion.terminology.pro_ctcae import PROCTCAELibrary


def _fold(text: str) -> str:
    d = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in d if unicodedata.category(ch) != "Mn")


# Functional/indirect complaints that must NOT be force-coded (spec section 2:
# "non riesco più a camminare" -> structured rationale + human review).
INDIRECT_REVIEW_CUES = [_fold(x) for x in (
    "non riesco", "non ce la faccio", "faccio fatica a", "non riesco piu",
)]


class BaselineMapper:
    def __init__(
        self,
        pro_library: PROCTCAELibrary,
        ctcae_dict: CTCAEDictionary | None = None,
        config: MappingConfig | None = None,
        normalizer: Normalizer | None = None,
        safety: SafetyRouter | None = None,
        calibrator: Calibrator | None = None,
    ):
        self.pro = pro_library
        self.ctcae = ctcae_dict
        self.config = config or MappingConfig()
        self.norm = normalizer or Normalizer()
        self.safety = safety or SafetyRouter()
        self.calibrator = calibrator or HeuristicCalibrator()

        # PRO index (respecting synthetic inclusion flag).
        pro_entries: list[IndexEntry] = []
        for e in self.pro.all_match_entries(include_synthetic=self.config.include_synthetic_terms):
            pro_entries.append(
                IndexEntry(
                    target_id=e.canonical_id,
                    term=self.pro.get(e.canonical_id).canonical_english,
                    surface=e.surface,
                    surface_match=self.norm.to_match(e.surface),
                    provenance=e.provenance,
                )
            )
        self.retriever = LexicalFuzzyRetriever(pro_entries, self.norm)

        # Exclusion index: (canonical_id, example_match).
        self._exclusions: list[tuple[str, str]] = []
        for term in self.pro:
            for ex in term.exclusion_examples:
                self._exclusions.append((term.canonical_id, self.norm.to_match(ex)))

        # CTCAE index (separate retriever).
        self.ctcae_retriever: LexicalFuzzyRetriever | None = None
        if self.ctcae is not None:
            ctcae_entries: list[IndexEntry] = []
            for t in self.ctcae:
                for surface, kind in t.match_entries():
                    ctcae_entries.append(
                        IndexEntry(
                            target_id=t.ctcae_id,
                            term=t.term,
                            surface=surface,
                            surface_match=self.norm.to_match(surface),
                            provenance=kind,
                        )
                    )
            self.ctcae_retriever = LexicalFuzzyRetriever(ctcae_entries, self.norm)

    # ------------------------------------------------------------------ #
    def _excluded_ids(self, seg_match: str) -> set[str]:
        hit: set[str] = set()
        th = self.config.thresholds.exclusion_fuzzy
        for cid, ex in self._exclusions:
            if not ex:
                continue
            if seg_match == ex or ex in seg_match or fuzzy_ratio(seg_match, ex) >= th:
                hit.add(cid)
        return hit

    def _pro_status(self, top1: float, top2: float) -> str:
        th = self.config.thresholds
        margin = top1 - top2
        if top1 >= th.tau_exact and margin >= th.margin_clear:
            return "EXACT_PRO_MATCH"
        if top1 >= th.tau_exact:
            return "MULTIPLE_POSSIBLE_PRO_MATCHES"
        if top1 >= th.tau_low and margin >= th.margin_clear:
            return "PRO_MATCH_WITH_AMBIGUITY"
        return "MULTIPLE_POSSIBLE_PRO_MATCHES"

    def _ctcae_fallback(self, display: str) -> CTCAEResult:
        if self.ctcae_retriever is None or self.ctcae is None:
            return CTCAEResult(version="6.0", status="NOT_EVALUATED", predictions=[], grade=None)
        cands = self.ctcae_retriever.retrieve(
            display, 0, top_k=3, min_score=0.6, fuzzy_floor=self.config.thresholds.fuzzy_floor
        )
        version = self.ctcae.version
        if not cands:
            return CTCAEResult(version=version, status="NO_MATCH", predictions=[], grade=None)
        th = self.config.thresholds
        best = cands[0]
        note = " (SYNTHETIC placeholder dictionary)" if self.ctcae.is_synthetic else ""
        preds = [
            CTCAEPrediction(
                ctcae_id=c.target_id,
                term=c.term,
                probability=self.calibrator.transform(c.score),
                evidence_spans=[display[c.evidence_start:c.evidence_end]],
                explanation=f"CTCAE fallback match via '{c.matched_surface}'{note}.",
            )
            for c in cands
            if c.score >= th.tau_low
        ]
        if not preds:
            return CTCAEResult(version=version, status="NO_MATCH", predictions=[], grade=None)
        status = "MATCH" if (len(preds) == 1 or best.score - cands[1].score >= th.margin_clear) else "AMBIGUOUS"
        # Grading is a separate, abstaining module (not run here): grade stays None.
        return CTCAEResult(version=version, status=status, predictions=preds, grade=None)

    # ------------------------------------------------------------------ #
    def map(self, request: MapRequest) -> MapResponse:
        th = self.config.thresholds
        display = self.norm.to_display(request.text)
        safety_dec = self.safety.check(display)

        meta: dict = {
            "calibration": {"method": self.calibrator.method, "is_calibrated": self.calibrator.is_calibrated},
            "retrieval_backend": "rapidfuzz" if _rapidfuzz_available() else "difflib",
            "thresholds": {"tau_exact": th.tau_exact, "tau_low": th.tau_low, "margin_clear": th.margin_clear},
            "safety_metalanguage": safety_dec.metalanguage_context,
            "segment_scores": [],
            "exclusion_triggered": False,
        }

        safety_result = SafetyResult(
            urgent_human_review=safety_dec.urgent_human_review,
            reason=safety_dec.reason,
        )

        if not display:
            return MapResponse(
                record_id=request.record_id,
                normalized_text=display,
                clinical_mentions=[],
                pro_ctcae=PROResult(status="INSUFFICIENT_CONTEXT", predictions=[]),
                ctcae=CTCAEResult(version="6.0", status="NOT_EVALUATED"),
                safety=safety_result,
                abstain=True,
                abstention_reason="Empty or non-informative input.",
                analysis_meta=meta,
            )

        segments = segment_text(display)
        mentions: list[ClinicalMention] = []
        surviving: dict[str, tuple[Candidate, object]] = {}  # cid -> (candidate, assertion)
        exclusion_hit = False

        for seg in segments:
            seg_match = self.norm.to_match(seg.text)
            excluded = self._excluded_ids(seg_match)
            if excluded:
                exclusion_hit = True
                meta["exclusion_triggered"] = True
            cands = self.retriever.retrieve(
                seg.text, seg.start, top_k=5, min_score=0.30, fuzzy_floor=th.fuzzy_floor
            )
            meta["segment_scores"].append(
                {"segment": seg.text, "top": [(c.target_id, round(c.score, 3)) for c in cands[:3]]}
            )
            for c in cands:
                if c.target_id in excluded:
                    continue
                if c.score < th.tau_low:
                    continue
                assertion = detect_assertion_temporality(display, c.evidence_start, c.evidence_end)
                # keep best score per canonical id
                if c.target_id not in surviving or c.score > surviving[c.target_id][0].score:
                    surviving[c.target_id] = (c, assertion)

        # Build clinical mentions from surviving candidates.
        for cid, (c, assertion) in surviving.items():
            mentions.append(
                ClinicalMention(
                    span=display[c.evidence_start:c.evidence_end],
                    start=c.evidence_start,
                    end=c.evidence_end,
                    assertion=assertion.assertion,
                    temporality=assertion.temporality,
                    experiencer=assertion.experiencer,
                )
            )

        # -------- PRO decision -------- #
        ranked = sorted(surviving.values(), key=lambda t: t[0].score, reverse=True)
        pro_status: str
        predictions: list[PROPrediction] = []
        abstain = False
        abstention_reason: str | None = None
        review_reason: str | None = None

        if ranked:
            best_c, best_a = ranked[0]
            top1 = best_c.score
            top2 = ranked[1][0].score if len(ranked) > 1 else 0.0
            if best_a.assertion == "negated":
                pro_status = "NEGATED_SYMPTOM"
            else:
                pro_status = self._pro_status(top1, top2)

            for c, a in ranked[:5]:
                term = self.pro.get(c.target_id)
                expl_bits = [f"matched '{c.matched_surface}' [{c.provenance}, {c.signals.get('signal')}]"]
                if a.assertion != "present":
                    expl_bits.append(f"assertion={a.assertion}")
                if a.temporality != "current":
                    expl_bits.append(f"temporality={a.temporality}")
                if a.experiencer != "patient":
                    expl_bits.append(f"experiencer={a.experiencer}")
                predictions.append(
                    PROPrediction(
                        canonical_id=c.target_id,
                        term=term.canonical_english,
                        probability=self.calibrator.transform(c.score),
                        evidence_spans=[display[c.evidence_start:c.evidence_end]],
                        applicable_attributes=list(term.attributes),
                        explanation="; ".join(expl_bits),
                    )
                )
        else:
            # No surviving PRO candidate.
            if exclusion_hit:
                pro_status = "OUT_OF_SCOPE"
            else:
                pro_status = "NO_DIRECT_PRO_MATCH"

        ctcae_result = CTCAEResult(version="6.0", status="NOT_EVALUATED")
        if pro_status in ("NO_DIRECT_PRO_MATCH", "OUT_OF_SCOPE"):
            ctcae_result = self._ctcae_fallback(display)

        # -------- indirect / functional complaint review -------- #
        folded = _fold(display)
        if pro_status == "NO_DIRECT_PRO_MATCH" and any(cue in folded for cue in INDIRECT_REVIEW_CUES):
            review_reason = (
                "Indirect / low-confidence functional complaint; not force-coded to a PRO "
                "term. Structured rationale generated and human review requested."
            )

        # -------- abstention logic (spec section 5, step 7) -------- #
        best_prob = predictions[0].probability if predictions else 0.0
        if pro_status == "EXACT_PRO_MATCH" and best_prob >= th.abstain_below:
            abstain = False
        elif pro_status == "NEGATED_SYMPTOM":
            abstain = False  # a definite negated determination is a valid answer
        elif pro_status in ("NO_DIRECT_PRO_MATCH", "OUT_OF_SCOPE"):
            if ctcae_result.status == "MATCH":
                abstain = False
            else:
                abstain = True
                abstention_reason = review_reason or "No direct PRO term and no confident CTCAE match."
        else:  # ambiguity / multiple / low confidence
            abstain = True
            abstention_reason = "Confidence below decision threshold; ambiguous candidates."

        if review_reason and not abstention_reason:
            abstain = True
            abstention_reason = review_reason
        if review_reason:
            meta["review_reason"] = review_reason
        meta["pro_status"] = pro_status
        meta["best_probability"] = best_prob

        return MapResponse(
            record_id=request.record_id,
            normalized_text=display,
            clinical_mentions=mentions,
            pro_ctcae=PROResult(status=pro_status, predictions=predictions),
            ctcae=ctcae_result,
            safety=safety_result,
            abstain=abstain,
            abstention_reason=abstention_reason,
            analysis_meta=meta,
        )


def _rapidfuzz_available() -> bool:
    try:
        import rapidfuzz  # noqa: F401

        return True
    except Exception:
        return False
