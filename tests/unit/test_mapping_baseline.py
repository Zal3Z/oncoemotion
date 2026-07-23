"""Structural properties of the baseline mapper output."""

from __future__ import annotations

from oncoemotion.schemas import MapRequest


def test_probabilities_in_unit_interval(mapper):
    r = mapper.map(MapRequest(record_id="p", text="ho un forte mal di testa"))
    for pred in r.pro_ctcae.predictions:
        assert 0.0 <= pred.probability <= 1.0


def test_predictions_sorted_descending(mapper):
    r = mapper.map(MapRequest(record_id="p", text="ho la diarrea e un po' di nausea"))
    probs = [p.probability for p in r.pro_ctcae.predictions]
    assert probs == sorted(probs, reverse=True)


def test_calibration_is_flagged_uncalibrated(mapper):
    r = mapper.map(MapRequest(record_id="p", text="ansia"))
    cal = r.analysis_meta["calibration"]
    # Phase 1 probabilities are explicitly uncalibrated heuristics.
    assert cal["is_calibrated"] is False
    assert cal["method"] == "uncalibrated_heuristic"


def test_ctcae_fallback_matches_fever(mapper):
    r = mapper.map(MapRequest(record_id="p", text="febbre"))
    assert r.ctcae.status == "MATCH"
    assert r.ctcae.predictions[0].term == "Fever"
    # If the official CTCAE is loaded it is NOT flagged synthetic; if the
    # placeholder is used, the explanation says so.
    expl = r.ctcae.predictions[0].explanation
    if mapper.ctcae.is_synthetic:
        assert "SYNTHETIC" in expl.upper()


def test_grade_not_auto_assigned(mapper):
    # Grading is a separate abstaining module; the mapper never sets a grade.
    r = mapper.map(MapRequest(record_id="p", text="febbre"))
    assert r.ctcae.grade is None


def test_empty_input_insufficient_context(mapper):
    r = mapper.map(MapRequest(record_id="p", text="   "))
    assert r.pro_ctcae.status == "INSUFFICIENT_CONTEXT"
    assert r.abstain is True


def test_applicable_attributes_present(mapper):
    r = mapper.map(MapRequest(record_id="p", text="ansia"))
    top = r.pro_ctcae.predictions[0]
    assert top.applicable_attributes == ["frequency", "severity", "interference"]
