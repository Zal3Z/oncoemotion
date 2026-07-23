"""Normalization, segmentation, assertion/temporality/experiencer."""

from __future__ import annotations

from oncoemotion.preprocessing.normalize import Normalizer
from oncoemotion.preprocessing.segment import segment_text
from oncoemotion.preprocessing.assertion import detect_assertion_temporality


def test_normalizer_display_and_match():
    n = Normalizer()
    assert n.to_display("  ciao   mondo ") == "ciao mondo"
    assert n.to_match("Perché È") == "perche e"  # lower + accent fold


def test_segmentation_multi_symptom():
    segs = segment_text("ho la diarrea e un po' di nausea")
    texts = [s.text for s in segs]
    assert "ho la diarrea" in texts
    assert any("nausea" in t for t in texts)
    # offsets map back into the original string
    for s in segs:
        assert "ho la diarrea e un po' di nausea"[s.start:s.end] == s.text


def test_negation_detection():
    text = "non ho nausea"
    start = text.index("nausea")
    res = detect_assertion_temporality(text, start, start + len("nausea"))
    assert res.assertion == "negated"


def test_resolved_temporality():
    text = "avevo nausea il mese scorso, ora è passata"
    start = text.index("nausea")
    res = detect_assertion_temporality(text, start, start + len("nausea"))
    assert res.temporality == "resolved"


def test_experiencer_other():
    text = "mia madre ha la tosse"
    start = text.index("tosse")
    res = detect_assertion_temporality(text, start, start + len("tosse"))
    assert res.experiencer == "other"


def test_present_current_default():
    text = "ho la tosse"
    start = text.index("tosse")
    res = detect_assertion_temporality(text, start, start + len("tosse"))
    assert res.assertion == "present"
    assert res.temporality == "current"
    assert res.experiencer == "patient"
