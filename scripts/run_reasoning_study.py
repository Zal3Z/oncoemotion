#!/usr/bin/env python
"""Run the joint-label direct/deliberative extension across the model cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from run_real_study import (
    _model_environment,
    _require_free_disk,
    _run,
    _safe_remove_model_cache,
    _slug,
)

_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reasoning_modes_for_model(config: dict, model_id: str) -> list[str]:
    modes = list(config["design"]["reasoning_modes"])
    native = config.get("native_reasoning", {}) or {}
    if native.get("enabled") and model_id in set(native.get("models", [])):
        modes.append(str(native.get("mode", "native_reasoning")))
    return list(dict.fromkeys(modes))


def _candidate_chunk_for_model(config: dict, model_id: str, default: int) -> int:
    """Return a memory-safe scoring batch without changing the scoring rule."""
    value = (
        config.get("classification", {})
        .get("candidate_chunk_by_model", {})
        .get(model_id, default)
    )
    value = int(value)
    if value < 1:
        raise ValueError(f"candidate chunk must be positive for {model_id}: {value}")
    return value


def _artifact_current(
    model_id: str,
    rows_path: Path,
    meta_path: Path,
    *,
    study_path: Path,
    dataset_path: Path,
    roles: list[str],
    reasoning_modes: list[str],
    expected_items: int,
    dtype: str,
    quantization: str | None,
) -> bool:
    if not all(path.exists() for path in (rows_path, meta_path, study_path, dataset_path)):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(
        [
            meta.get("model_id") == model_id,
            meta.get("dtype") == dtype,
            meta.get("quantization") == quantization,
            meta.get("study_config_sha256") == _sha256(study_path),
            meta.get("dataset_sha256") == _sha256(dataset_path),
            meta.get("rows_sha256") == _sha256(rows_path),
            meta.get("roles") == roles,
            meta.get("reasoning_modes") == reasoning_modes,
            meta.get("decision_space") == "joint",
            meta.get("explicit_nonclassifiable") is True,
            meta.get("n_items") == expected_items,
            meta.get("n_rows") == expected_items * len(roles) * len(reasoning_modes),
            meta.get("text_redacted") is True,
            meta.get("reasoning_text_redacted") is True,
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument(
        "--study-config",
        type=Path,
        default=_ROOT / "configs/study_esmo_2026_reasoning.yaml",
    )
    ap.add_argument("--cohort", choices=["primary", "all"], default="primary")
    ap.add_argument("--models", nargs="+", help="pilot override; pooled analysis is skipped")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quantization", choices=["nf4", "int8"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reasoning-max-new-tokens", type=int, default=0)
    ap.add_argument("--native-reasoning-max-new-tokens", type=int, default=0)
    ap.add_argument("--candidate-chunk", type=int, default=32)
    ap.add_argument("--ephemeral-model-cache-root", type=Path)
    ap.add_argument("--min-free-disk-gb", type=float, default=0.0)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "data/real/clinical_real.jsonl",
    )
    ap.add_argument(
        "--outroot",
        type=Path,
        default=_ROOT / "outputs/reasoning_real",
    )
    args = ap.parse_args()

    config = yaml.safe_load(args.study_config.read_text(encoding="utf-8")) or {}
    primary = list(config.get("models", {}).get("primary", []))
    secondary = list(config.get("models", {}).get("secondary", []))
    configured = primary if args.cohort == "primary" else [*primary, *secondary]
    models = list(args.models or configured)
    design = config["design"]
    roles = list(design["roles"])
    expected_items = int(design["expected_records_per_model"])
    reasoning_max = int(
        args.reasoning_max_new_tokens
        or config.get("classification", {}).get("reasoning_max_new_tokens", 80)
    )

    _run(
        [
            PY,
            str(_ROOT / "scripts/ingest_real_fields.py"),
            "--xlsx",
            str(args.xlsx),
            "--out",
            str(args.dataset),
            "--expected-records",
            str(expected_items),
        ]
    )
    args.outroot.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    for model_id in models:
        modes = _reasoning_modes_for_model(config, model_id)
        candidate_chunk = _candidate_chunk_for_model(
            config, model_id, args.candidate_chunk
        )
        slug = _slug(model_id)
        rows_path = args.outroot / f"{slug}__rows.jsonl"
        meta_path = args.outroot / f"{slug}__meta.json"
        if (
            args.skip_existing
            and not args.limit
            and _artifact_current(
                model_id,
                rows_path,
                meta_path,
                study_path=args.study_config,
                dataset_path=args.dataset,
                roles=roles,
                reasoning_modes=modes,
                expected_items=expected_items,
                dtype=args.dtype,
                quantization=args.quantization,
            )
        ):
            print(f"[skip] {model_id}: current reasoning artifact", flush=True)
            completed.append(rows_path)
            continue

        cache_dir = None
        child_env = None
        if args.ephemeral_model_cache_root:
            cache_dir, child_env = _model_environment(args.ephemeral_model_cache_root, model_id)
        try:
            if args.min_free_disk_gb:
                _require_free_disk(
                    args.ephemeral_model_cache_root or args.outroot,
                    args.min_free_disk_gb,
                )
            command = [
                PY,
                "-u",
                str(_ROOT / "scripts/run_reasoning_classification.py"),
                "--model",
                model_id,
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--dataset",
                str(args.dataset),
                "--out",
                str(args.outroot),
                "--study-config",
                str(args.study_config),
                "--roles",
                *roles,
                "--reasoning-modes",
                *modes,
                "--reasoning-max-new-tokens",
                str(reasoning_max),
                "--native-reasoning-max-new-tokens",
                str(
                    args.native_reasoning_max_new_tokens
                    or config.get("native_reasoning", {}).get("max_new_tokens", 512)
                ),
                "--candidate-chunk",
                str(candidate_chunk),
                "--redact-text",
            ]
            if args.quantization:
                command.extend(["--quantization", args.quantization])
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            _run(command, env=child_env)
            completed.append(rows_path)
        finally:
            if cache_dir is not None:
                _safe_remove_model_cache(cache_dir, args.ephemeral_model_cache_root)
                print(f"[cache cleared] {model_id}", flush=True)

    if args.limit or args.models:
        print("Pilot/override complete; pooled reasoning analysis intentionally skipped.")
        return 0

    _run(
        [
            PY,
            str(_ROOT / "scripts/analyze_reasoning_classification.py"),
            *[str(path) for path in completed],
            "--study-config",
            str(args.study_config),
            "--cohort",
            args.cohort,
            "--out",
            str(args.outroot / "reasoning_analysis.json"),
        ]
    )
    print("Reasoning/abstention extension complete; only redacted artifacts may be exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
