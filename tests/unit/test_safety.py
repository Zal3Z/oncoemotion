"""Independent safety routing."""

from __future__ import annotations

from oncoemotion.safety.router import SafetyRouter


def test_suicide_flagged_urgent():
    d = SafetyRouter().check("suicidio")
    assert d.urgent_human_review is True
    assert d.matched_cues


def test_english_self_harm_flagged():
    d = SafetyRouter().check("I want to kill myself")
    assert d.urgent_human_review is True


def test_benign_not_flagged():
    d = SafetyRouter().check("ho un po' di nausea")
    assert d.urgent_human_review is False


def test_metalanguage_context_recorded():
    d = SafetyRouter().check("la parola da classificare è: suicidio")
    assert d.urgent_human_review is True   # still routed to review (conservative)
    assert d.metalanguage_context is True


def test_extra_cues_configurable():
    d = SafetyRouter(extra_cues=["voglio sparire per sempre"]).check("voglio sparire per sempre")
    assert d.urgent_human_review is True
