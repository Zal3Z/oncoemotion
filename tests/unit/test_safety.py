"""Independent safety routing."""

from __future__ import annotations

import pytest

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


# Phrasings that slipped through until the expanded item set exposed them: an
# inflected form the stem did not cover, a nominalized desire to die, and
# hopelessness expressed without any death word.
@pytest.mark.parametrize("text", [
    "Pensieri autolesivi ricorrenti riferiti negli ultimi giorni.",
    "Il desiderio di morire e costante nelle ultime settimane.",
    "Non vedo piu una via d'uscita in modo definitivo.",
    "Ho una voglia di morire che non passa.",
    "Continuo ad avere pensieri di morte.",
])
def test_previously_missed_phrasings_are_flagged(text):
    assert SafetyRouter().check(text).urgent_human_review is True


@pytest.mark.parametrize("text", [
    "Il parcheggio dell'ospedale era pieno anche stamattina.",
    "Stanchezza costante che limita le normali attivita domestiche.",
    "Ho comprato una crema nuova per il viso in farmacia.",
])
def test_benign_clinical_and_admin_text_not_flagged(text):
    assert SafetyRouter().check(text).urgent_human_review is False
