"""FastAPI app for the mapper + research endpoints (spec section 15).

Endpoints:
    POST /map                  clinical mapping — NEVER performs steering
    POST /analyze-activations  research-only; disabled unless ML stack + flag
    POST /run-steering         research-only; DISABLED by default, never in prod
    GET  /terminology/pro-ctcae
    GET  /health

Privacy: request free-text is never written to logs (spec section 17); only the
pseudonymous record_id and the resulting status are logged.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from oncoemotion import __version__
from oncoemotion.factory import build_default_mapper
from oncoemotion.schemas import MapRequest, MapResponse

logger = logging.getLogger("oncoemotion.api")


def _enabled(env: str) -> bool:
    return os.environ.get(env, "0").lower() in ("1", "true", "yes", "on")


class SteeringRequest(BaseModel):
    record_id: str
    text: str
    layer: int
    vector_name: str
    alpha: float


def create_app(mapper=None) -> FastAPI:
    app = FastAPI(
        title="oncoemotion — PRO-CTCAE/CTCAE mapping + emotion-concept research API",
        version=__version__,
        description=(
            "Research and clinician-support tool. Emotion scores are 'emotion-like "
            "internal representations', not diagnoses or conscious emotions. Not for "
            "autonomous diagnosis; does not replace clinical review."
        ),
    )
    app.state.mapper = mapper or build_default_mapper()

    @app.get("/health")
    def health() -> dict:
        m = app.state.mapper
        return {
            "status": "ok",
            "version": __version__,
            "pro_terms": len(m.pro),
            "ctcae_loaded": m.ctcae is not None,
            "ctcae_synthetic": bool(m.ctcae and m.ctcae.is_synthetic),
            "steering_enabled": _enabled("ONCOEMOTION_ENABLE_STEERING"),
        }

    @app.post("/map", response_model=MapResponse)
    def map_record(req: MapRequest) -> MapResponse:
        resp = app.state.mapper.map(req)
        # PII-safe log line: no free text.
        logger.info("map record_id=%s pro_status=%s abstain=%s",
                    req.record_id, resp.pro_ctcae.status, resp.abstain)
        return resp

    @app.get("/terminology/pro-ctcae")
    def terminology() -> dict:
        m = app.state.mapper
        return {
            "count": len(m.pro),
            "metadata": m.pro.metadata,
            "terms": [
                {
                    "canonical_id": t.canonical_id,
                    "canonical_english": t.canonical_english,
                    "category": t.category,
                    "attributes": t.attributes,
                    "official_italian_labels": t.official_italian_labels,
                }
                for t in m.pro
            ],
        }

    @app.post("/analyze-activations")
    def analyze_activations(req: MapRequest) -> dict:
        if not _enabled("ONCOEMOTION_ENABLE_ACTIVATIONS"):
            raise HTTPException(
                status_code=501,
                detail=("Activation analysis is a research-only feature and requires the "
                        "ML stack (Phase 2+). Set ONCOEMOTION_ENABLE_ACTIVATIONS=1 to enable."),
            )
        raise HTTPException(status_code=501, detail="Not implemented in this build (Phase 2+).")

    @app.post("/run-steering")
    def run_steering(req: SteeringRequest) -> dict:
        # Research-only; disabled by default; never exposed in production.
        if not _enabled("ONCOEMOTION_ENABLE_STEERING"):
            raise HTTPException(
                status_code=403,
                detail=("Steering endpoint is research-only and disabled by default. "
                        "Enable explicitly with ONCOEMOTION_ENABLE_STEERING=1 in a research "
                        "environment. It must never be exposed in production."),
            )
        # Audit trail: record what would be applied (model/layer/vector/alpha).
        logger.warning("STEERING request record_id=%s layer=%s vector=%s alpha=%s",
                       req.record_id, req.layer, req.vector_name, req.alpha)
        raise HTTPException(status_code=501, detail="Steering runtime lands in Phase 4.")

    return app


app = None  # lazily created by ASGI servers via create_app()
