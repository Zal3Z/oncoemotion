#!/usr/bin/env python
"""Run the clinician-validated real-field study end to end.

The command ingests the private workbook locally, builds model-specific emotion
directions from the independent synthetic dataset, measures two token-matched role
conditions on real text with redacted outputs, and runs the dedicated analysis.
Raw workbook text and the ingested JSONL are never included in exported results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _role_artifact_current(
    model_id: str,
    rows_path: Path,
    meta_path: Path,
    *,
    study_path: Path,
    dataset_path: Path,
    vectors_path: Path,
    validation_path: Path,
    roles: list[str],
    expected_items: int,
) -> bool:
    if not rows_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all([
        meta.get("model_id") == model_id,
        meta.get("study_config_sha256") == _sha256(study_path),
        meta.get("dataset_sha256") == _sha256(dataset_path),
        meta.get("vectors_sha256") == _sha256(vectors_path),
        meta.get("validation_sha256") == _sha256(validation_path),
        meta.get("roles") == roles,
        meta.get("n_items") == expected_items,
        meta.get("n_rows") == expected_items * len(roles),
        meta.get("arms") == ["intact"],
        meta.get("arms_requested") == ["intact"],
        meta.get("scorer") == "adaptive",
        meta.get("text_redacted") is True,
        meta.get("rows_sha256") == _sha256(rows_path),
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument(
        "--study-config",
        type=Path,
        default=_ROOT / "configs/study_esmo_2026_real.yaml",
    )
    ap.add_argument("--cohort", choices=["primary", "all"], default="primary")
    ap.add_argument("--models", nargs="+", help="pilot override; disables complete-cohort analysis")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="pilot records per model; 0 = full source")
    ap.add_argument("--baseline-limit", type=int, default=0)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "data/real/clinical_real.jsonl",
    )
    ap.add_argument(
        "--model-outroot",
        type=Path,
        default=_ROOT / "outputs/models_real",
    )
    ap.add_argument(
        "--outroot",
        type=Path,
        default=_ROOT / "outputs/real_fields",
    )
    args = ap.parse_args()

    config = yaml.safe_load(args.study_config.read_text(encoding="utf-8")) or {}
    primary = list(config.get("models", {}).get("primary", []))
    secondary = list(config.get("models", {}).get("secondary", []))
    configured = primary if args.cohort == "primary" else [*primary, *secondary]
    models = list(args.models or configured)
    roles = list(config["design"]["roles"])
    expected_records = int(config["design"]["expected_records_per_model"])

    ingest = [
        PY,
        str(_ROOT / "scripts/ingest_real_fields.py"),
        "--xlsx",
        str(args.xlsx),
        "--out",
        str(args.dataset),
        "--expected-records",
        str(expected_records),
    ]
    _run(ingest)

    vector_command = [
        PY,
        str(_ROOT / "scripts/run_all_models.py"),
        "--models",
        *models,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--outroot",
        str(args.model_outroot),
        "--stages",
        "vectors",
        "--methods",
        "diff_of_means",
        "--study-config",
        str(args.study_config),
    ]
    if args.skip_existing:
        vector_command.append("--skip-existing")
    _run(vector_command)

    args.outroot.mkdir(parents=True, exist_ok=True)
    completed_rows = []
    for model_id in models:
        slug = _slug(model_id)
        model_dir = args.model_outroot / slug
        vectors = model_dir / "emotion_vectors.npz"
        validation = model_dir / "vector_validation.json"
        rows_path = args.outroot / f"{slug}__rows.jsonl"
        meta_path = args.outroot / f"{slug}__meta.json"
        if args.skip_existing and not args.limit and _role_artifact_current(
            model_id,
            rows_path,
            meta_path,
            study_path=args.study_config,
            dataset_path=args.dataset,
            vectors_path=vectors,
            validation_path=validation,
            roles=roles,
            expected_items=expected_records,
        ):
            print(f"[skip] {model_id}: current real-field artifact")
            completed_rows.append(rows_path)
            continue
        command = [
            PY,
            str(_ROOT / "scripts/run_role_emotion.py"),
            "--model",
            model_id,
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--dataset",
            str(args.dataset),
            "--vecs",
            str(vectors),
            "--val-report",
            str(validation),
            "--out",
            str(args.outroot),
            "--roles",
            *roles,
            "--arms",
            "intact",
            "--scorer",
            "adaptive",
            "--study-config",
            str(args.study_config),
            "--redact-text",
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        if args.baseline_limit:
            command.extend(["--baseline-limit", str(args.baseline_limit)])
        _run(command)
        completed_rows.append(rows_path)

    if args.limit or args.models:
        print("\nPilot/override complete; pooled confirmatory analysis intentionally skipped.")
        return 0

    analysis_command = [
        PY,
        str(_ROOT / "scripts/analyze_real_fields.py"),
        *[str(path) for path in completed_rows],
        "--study-config",
        str(args.study_config),
        "--cohort",
        args.cohort,
        "--out",
        str(args.outroot / "real_primary_analysis.json"),
    ]
    _run(analysis_command)
    print("\nReal-field study complete. Export outputs only; keep the workbook and data/real local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
