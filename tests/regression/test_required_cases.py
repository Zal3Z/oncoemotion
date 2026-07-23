"""Mandatory behavioural cases from spec sections 2 and 16."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oncoemotion.schemas import MapRequest, MapResponse

_ROOT = Path(__file__).resolve().parents[2]


def _map(mapper, text: str) -> MapResponse:
    return mapper.map(MapRequest(record_id="t", text=text))


def _pred_ids(resp: MapResponse) -> list[str]:
    return [p.canonical_id for p in resp.pro_ctcae.predictions]


def test_ansia_maps_to_anxious(mapper):
    r = _map(mapper, "ansia")
    assert r.pro_ctcae.status == "EXACT_PRO_MATCH"
    assert _pred_ids(r)[0] == "PRO_054"


def test_nail_discoloration_from_patient_phrase(mapper):
    r = _map(mapper, "mi si ingialliscono le unghie")
    assert r.pro_ctcae.status == "EXACT_PRO_MATCH"
    assert _pred_ids(r)[0] == "PRO_033"


def test_febbre_no_direct_pro_but_ctcae_fever(mapper):
    r = _map(mapper, "febbre")
    assert r.pro_ctcae.status == "NO_DIRECT_PRO_MATCH"
    assert r.ctcae.status == "MATCH"
    assert r.ctcae.predictions and r.ctcae.predictions[0].term == "Fever"


def test_suicidio_urgent_human_review(mapper):
    r = _map(mapper, "suicidio")
    assert r.pro_ctcae.status == "NO_DIRECT_PRO_MATCH"
    assert r.safety.urgent_human_review is True
    assert r.safety.reason


def test_cannot_walk_not_force_coded(mapper):
    r = _map(mapper, "non riesco più a camminare")
    assert r.pro_ctcae.status == "NO_DIRECT_PRO_MATCH"
    # must NOT be auto-assigned to Fatigue / Muscle pain / General pain / Joint pain
    forbidden = {"PRO_053", "PRO_050", "PRO_048", "PRO_051"}
    assert not (set(_pred_ids(r)) & forbidden)
    assert r.abstain is True
    assert r.abstention_reason


def test_no_nausea_is_negated(mapper):
    r = _map(mapper, "non ho nausea")
    assert r.pro_ctcae.status == "NEGATED_SYMPTOM"
    nausea_mentions = [m for m in r.clinical_mentions if "nausea" in m.span.lower()]
    assert nausea_mentions and nausea_mentions[0].assertion == "negated"


def test_past_resolved_nausea_not_current(mapper):
    r = _map(mapper, "avevo nausea il mese scorso, ora è passata")
    mentions = [m for m in r.clinical_mentions if "nausea" in m.span.lower()]
    assert mentions
    assert mentions[0].temporality == "resolved"
    assert all(m.temporality != "current" for m in mentions)


def test_nail_polish_not_clinical_event(mapper):
    r = _map(mapper, "ho messo lo smalto giallo")
    assert "PRO_033" not in _pred_ids(r)
    assert r.pro_ctcae.status in ("OUT_OF_SCOPE", "NO_DIRECT_PRO_MATCH")


def test_determinism_same_input(mapper):
    a = _map(mapper, "ho un forte mal di testa da tre giorni").model_dump()
    b = _map(mapper, "ho un forte mal di testa da tre giorni").model_dump()
    assert a == b


def test_batch_order_invariance(mapper):
    texts = ["ansia", "febbre", "non ho nausea", "mi si ingialliscono le unghie"]
    forward = [_map(mapper, t).model_dump() for t in texts]
    backward = [_map(mapper, t).model_dump() for t in reversed(texts)]
    backward.reverse()
    assert forward == backward


@pytest.mark.parametrize("line", _ROOT.joinpath("data/synthetic/clinical_controls.jsonl").read_text(encoding="utf-8").splitlines())
def test_schema_always_valid(mapper, line):
    line = line.strip()
    if not line:
        return
    rec = json.loads(line)
    resp = mapper.map(MapRequest(**rec))
    # Round-trip through the schema: always valid.
    dumped = resp.model_dump()
    MapResponse.model_validate(dumped)
