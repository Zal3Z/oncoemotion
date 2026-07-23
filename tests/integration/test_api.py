"""API integration tests (spec sections 15, 16, 17)."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from oncoemotion.api.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["pro_terms"] == 80
    assert body["steering_enabled"] is False  # disabled by default


def test_terminology_endpoint(client):
    r = client.get("/terminology/pro-ctcae")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 80
    assert len(body["terms"]) == 80


def test_map_endpoint_returns_valid_response(client):
    r = client.post("/map", json={"record_id": "x1", "text": "ansia", "language": "it"})
    assert r.status_code == 200
    body = r.json()
    assert body["pro_ctcae"]["status"] == "EXACT_PRO_MATCH"
    assert body["pro_ctcae"]["predictions"][0]["canonical_id"] == "PRO_054"


def test_map_endpoint_does_not_steer(client):
    # The clinical endpoint must never run steering; analysis_meta carries no
    # steering fields and the response is a plain mapping.
    r = client.post("/map", json={"record_id": "x2", "text": "febbre"})
    assert r.status_code == 200
    assert "steering" not in (r.json().get("analysis_meta") or {})


def test_steering_disabled_by_default(client):
    r = client.post(
        "/run-steering",
        json={"record_id": "x", "text": "ansia", "layer": 12, "vector_name": "afraid", "alpha": 0.05},
    )
    assert r.status_code == 403  # research-only, disabled


def test_analyze_activations_requires_ml(client):
    r = client.post("/analyze-activations", json={"record_id": "x", "text": "ansia"})
    assert r.status_code == 501


def test_no_free_text_in_logs(client, caplog):
    marker = "ZZUNIQUEPATIENTPHRASE42"
    with caplog.at_level(logging.INFO, logger="oncoemotion.api"):
        client.post("/map", json={"record_id": "pii-check", "text": f"ansia {marker}"})
    # The pseudonymous record_id may appear; the free text must NOT.
    assert marker not in caplog.text
    assert "pii-check" in caplog.text
