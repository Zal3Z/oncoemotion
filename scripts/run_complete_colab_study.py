#!/usr/bin/env python
"""Run the complete ESMO workflow one model at a time.

The orchestrator keeps a model in one isolated Hugging Face cache while it runs
the controlled synthetic study, the validated real-field v1 study, and the joint
direct/deliberative extension.  The cache is removed only after all requested
phases for that model have completed.  Scientific outputs remain resumable and
are accepted only when the underlying scripts' fingerprints still match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PRIMARY_PHASES = ("controlled", "real", "reasoning")
DEFAULT_RUNTIME_CONFIG = _ROOT / "configs/runtime_blackwell_96gb.yaml"


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _runtime_for_model(runtime_config: dict, model_id: str) -> dict:
    """Resolve and validate the declared numerical representation for one model."""
    resolved = dict(runtime_config.get("default", {}) or {})
    resolved.update((runtime_config.get("models", {}) or {}).get(model_id, {}) or {})
    quantization = resolved.get("quantization")
    if quantization not in {None, "nf4", "int8"}:
        raise ValueError(f"unsupported quantization for {model_id}: {quantization!r}")
    resolved["dtype"] = str(resolved.get("dtype", "bfloat16"))
    resolved["quantization"] = quantization
    resolved["minimum_free_disk_gb"] = float(
        resolved.get("minimum_free_disk_gb", 0.0)
    )
    return resolved


def _run(command: list[str], *, env: dict[str, str] | None = None, label: str) -> None:
    started = time.time()
    print(f"  > {label}", flush=True)
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=_ROOT,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if line.strip():
            print(f"    | {line.rstrip()}", flush=True)
    return_code = process.wait()
    elapsed = time.time() - started
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"  < OK {label} ({elapsed / 60:.1f} min)", flush=True)


def _model_environment(cache_root: Path, model_id: str) -> tuple[Path, dict[str, str]]:
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    model_root = (cache_root / _slug(model_id)).resolve()
    if model_root.parent != cache_root:
        raise ValueError(f"unsafe model cache path: {model_root}")
    model_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HOME"] = str(model_root)
    env["HF_HUB_CACHE"] = str(model_root / "hub")
    env["HF_XET_CACHE"] = str(model_root / "xet")
    env["HF_ASSETS_CACHE"] = str(model_root / "assets")
    env["TRANSFORMERS_CACHE"] = str(model_root / "transformers")
    env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    env["HF_XET_SHARD_CACHE_SIZE_LIMIT"] = "1000000000"
    return model_root, env


def _remove_model_cache(model_root: Path, cache_root: Path) -> None:
    root = cache_root.resolve()
    target = model_root.resolve()
    if target.parent != root or target == root:
        raise ValueError(f"refusing to remove unsafe cache path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1_000_000_000


def _controlled_current(
    model_id: str,
    config: Path,
    dataset: Path,
    *,
    dtype: str,
    quantization: str | None,
) -> bool:
    slug = _slug(model_id)
    rows = _ROOT / "outputs/role_emotion" / f"{slug}__rows.jsonl"
    meta_path = rows.with_name(f"{slug}__meta.json")
    vectors = _ROOT / "outputs/models" / slug / "emotion_vectors.npz"
    validation = _ROOT / "outputs/models" / slug / "vector_validation.json"
    meta = _read_json(meta_path)
    study = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    if not all(path.is_file() for path in (rows, vectors, validation, config, dataset)):
        return False
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        commit = None
    return all(
        (
            meta.get("protocol_id") == study.get("protocol_id"),
            meta.get("model_id") == model_id,
            meta.get("dtype") == dtype,
            meta.get("quantization") == quantization,
            meta.get("git_commit") == commit,
            meta.get("study_config_sha256") == _sha256(config),
            meta.get("dataset_sha256") == _sha256(dataset),
            meta.get("vectors_sha256") == _sha256(vectors),
            meta.get("validation_sha256") == _sha256(validation),
            meta.get("rows_sha256") == _sha256(rows),
            meta.get("roles") == ["oncologo", "generico", "none_task", "none_filler"],
            meta.get("arms_requested") == ["intact", "emotion", "random"],
            meta.get("scorer") == "both",
            meta.get("ablation_limit") == 60,
        )
    )


def _configured_model_sets(cohort: str) -> dict[str, list[str]]:
    controlled = yaml.safe_load(
        (_ROOT / "configs/study_esmo_2026.yaml").read_text(encoding="utf-8")
    )
    real = yaml.safe_load(
        (_ROOT / "configs/study_esmo_2026_real.yaml").read_text(encoding="utf-8")
    )
    reasoning = yaml.safe_load(
        (_ROOT / "configs/study_esmo_2026_reasoning.yaml").read_text(encoding="utf-8")
    )
    controlled_primary = list(controlled["models"]["tier1"])
    real_primary = list(real["models"]["primary"])
    if real_primary != controlled_primary:
        raise ValueError("controlled and real primary model cohorts differ")
    return {
        "controlled": controlled_primary,
        "real": [
            *real_primary,
            *(list(real["models"].get("secondary", [])) if cohort == "all" else []),
        ],
        "reasoning": [
            *list(reasoning["models"]["primary"]),
            *(
                list(reasoning["models"].get("secondary", []))
                if cohort == "all"
                else []
            ),
        ],
    }


def _run_controlled(
    model_id: str,
    env: dict[str, str],
    *,
    force: bool,
    dtype: str,
    quantization: str | None,
) -> None:
    config = _ROOT / "configs/study_esmo_2026.yaml"
    dataset = _ROOT / "data/synthetic/clinical_labeled.jsonl"
    vector_cmd = [
        PY,
        "-u",
        str(_ROOT / "scripts/run_all_models.py"),
        "--models",
        model_id,
        "--dtype",
        dtype,
        "--device",
        "auto",
        "--stages",
        "vectors",
        "--study-config",
        str(config),
    ]
    if quantization:
        vector_cmd.extend(["--quantization", quantization])
    if not force:
        vector_cmd.append("--skip-existing")
    _run(vector_cmd, env=env, label="vettori emozionali controllati")
    slug = _slug(model_id)
    vectors = _ROOT / "outputs/models" / slug / "emotion_vectors.npz"
    validation = _ROOT / "outputs/models" / slug / "vector_validation.json"
    if force or not _controlled_current(
        model_id,
        config,
        dataset,
        dtype=dtype,
        quantization=quantization,
    ):
        role_cmd = [
                PY,
                "-u",
                str(_ROOT / "scripts/run_role_emotion.py"),
                "--model",
                model_id,
                "--dtype",
                dtype,
                "--device",
                "auto",
                "--roles",
                "oncologo",
                "generico",
                "none_task",
                "none_filler",
                "--arms",
                "intact",
                "emotion",
                "random",
                "--scorer",
                "both",
                "--study-config",
                str(config),
                "--dataset",
                str(dataset),
                "--ablation-limit",
                "60",
                "--vecs",
                str(vectors),
                "--val-report",
                str(validation),
            ]
        if quantization:
            role_cmd.extend(["--quantization", quantization])
        _run(role_cmd, env=env, label="studio controllato ruolo x affetto")
    else:
        print("  [skip] studio controllato già completo e compatibile", flush=True)
    _run(
        [
            PY,
            str(_ROOT / "scripts/analyze_role_emotion.py"),
            f"--rows={_ROOT / 'outputs/role_emotion' / f'{slug}__rows.jsonl'}",
        ],
        env=env,
        label="analisi controllata per modello",
    )


def _run_real(
    model_id: str,
    xlsx: Path,
    env: dict[str, str],
    *,
    force: bool,
    dtype: str,
    quantization: str | None,
) -> None:
    command = [
        PY,
        "-u",
        str(_ROOT / "scripts/run_real_study.py"),
        "--xlsx",
        str(xlsx),
        "--models",
        model_id,
        "--dtype",
        dtype,
        "--device",
        "auto",
    ]
    if quantization:
        command.extend(["--quantization", quantization])
    if force:
        command.append("--no-skip-existing")
    _run(command, env=env, label="campi reali validati v1")


def _run_reasoning(
    model_id: str,
    xlsx: Path,
    env: dict[str, str],
    *,
    force: bool,
    reasoning_max_new_tokens: int,
    native_reasoning_max_new_tokens: int,
    dtype: str,
    quantization: str | None,
) -> None:
    command = [
        PY,
        "-u",
        str(_ROOT / "scripts/run_reasoning_study.py"),
        "--xlsx",
        str(xlsx),
        "--models",
        model_id,
        "--dtype",
        dtype,
        "--device",
        "auto",
        "--reasoning-max-new-tokens",
        str(reasoning_max_new_tokens),
        "--native-reasoning-max-new-tokens",
        str(native_reasoning_max_new_tokens),
    ]
    if quantization:
        command.extend(["--quantization", quantization])
    if force:
        command.append("--no-skip-existing")
    _run(command, env=env, label="PRO/CTCAE/astensione direct-deliberative")


def _run_aggregate_analysis(phases: list[str], cohort: str) -> None:
    panels = _configured_model_sets(cohort)
    if "controlled" in phases:
        _run(
            [
                PY,
                str(_ROOT / "scripts/analyze_results.py"),
                "--rows-glob",
                str(_ROOT / "outputs/role_emotion/*__rows.jsonl"),
                "--study-config",
                str(_ROOT / "configs/study_esmo_2026.yaml"),
            ],
            label="analisi ESMO controllata aggregata",
        )
        _run(
            [PY, str(_ROOT / "scripts/build_esmo_abstract.py")],
            label="bozza abstract ESMO",
        )
        _run(
            [PY, str(_ROOT / "scripts/build_esmo_poster_figures.py")],
            label="figure poster ESMO",
        )
    if "real" in phases:
        real_rows = [
            str(_ROOT / "outputs/real_fields" / f"{_slug(model)}__rows.jsonl")
            for model in panels["real"]
        ]
        _run(
            [
                PY,
                str(_ROOT / "scripts/analyze_real_fields.py"),
                *real_rows,
                "--study-config",
                str(_ROOT / "configs/study_esmo_2026_real.yaml"),
                "--cohort",
                cohort,
                "--out",
                str(_ROOT / "outputs/real_fields/real_primary_analysis.json"),
            ],
            label="analisi aggregata campi reali",
        )
        _run(
            [
                PY,
                str(_ROOT / "scripts/package_real_results.py"),
                "--out",
                str(_ROOT / "outputs/packages/oncoemotion_real_results.zip"),
            ],
            label="pacchetto redatto campi reali",
        )
    if "reasoning" in phases:
        reasoning_rows = [
            str(_ROOT / "outputs/reasoning_real" / f"{_slug(model)}__rows.jsonl")
            for model in panels["reasoning"]
        ]
        _run(
            [
                PY,
                str(_ROOT / "scripts/analyze_reasoning_classification.py"),
                *reasoning_rows,
                "--study-config",
                str(_ROOT / "configs/study_esmo_2026_reasoning.yaml"),
                "--cohort",
                cohort,
                "--out",
                str(_ROOT / "outputs/reasoning_real/reasoning_analysis.json"),
            ],
            label="analisi aggregata reasoning/astensione",
        )
        _run(
            [
                PY,
                str(_ROOT / "scripts/package_reasoning_results.py"),
                "--out",
                str(_ROOT / "outputs/packages/oncoemotion_reasoning_results.zip"),
            ],
            label="pacchetto redatto reasoning/astensione",
        )
    if "real" in phases:
        audit_cmd = [
            PY,
            str(_ROOT / "scripts/export_item_audit.py"),
            "--results-dir",
            str(_ROOT / "outputs/real_fields"),
            "--dataset",
            str(_ROOT / "data/real/clinical_real.jsonl"),
            "--outdir",
            str(_ROOT / "outputs/tables/item_audit"),
        ]
        if "reasoning" in phases:
            audit_cmd.extend(
                [
                    "--reasoning-results-dir",
                    str(_ROOT / "outputs/reasoning_real"),
                ]
            )
        _run(audit_cmd, label="tabelle locali di audit per item")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PRIMARY_PHASES,
        default=list(PRIMARY_PHASES),
    )
    parser.add_argument("--cohort", choices=["primary", "all"], default="primary")
    parser.add_argument("--models", nargs="+", help="optional primary pilot override")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/content/oncoemotion_hf_full_cache"),
    )
    parser.add_argument("--keep-model-cache", action="store_true")
    parser.add_argument("--min-free-disk-gb", type=float, default=75.0)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG,
        help="per-model dtype/quantization profile for the selected accelerator",
    )
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=80)
    parser.add_argument("--native-reasoning-max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    args.xlsx = args.xlsx.resolve()
    if not args.xlsx.is_file():
        raise FileNotFoundError(args.xlsx)
    phases = list(dict.fromkeys(args.phases))
    runtime_config = yaml.safe_load(
        args.runtime_config.read_text(encoding="utf-8")
    ) or {}
    panels = _configured_model_sets(args.cohort)
    configured_union = list(
        dict.fromkeys(
            [
                *panels["controlled"],
                *panels["real"],
                *panels["reasoning"],
            ]
        )
    )
    models = list(args.models or configured_union)
    runtime_by_model = {
        model_id: _runtime_for_model(runtime_config, model_id) for model_id in models
    }

    _run(
        [PY, str(_ROOT / "scripts/generate_labeled_clinical.py"), "--check-only"],
        label="validazione dataset sintetico controllato",
    )
    _run(
        [
            PY,
            str(_ROOT / "scripts/ingest_real_fields.py"),
            "--xlsx",
            str(args.xlsx),
            "--out",
            str(_ROOT / "data/real/clinical_real.jsonl"),
            "--expected-records",
            "1275",
        ],
        label="ingestione dataset reale validato",
    )

    started = time.time()
    print(
        f"\nRun completo: {len(models)} modelli | fasi={phases} | "
        f"coorte={args.cohort} | force={args.force}",
        flush=True,
    )
    for index, model_id in enumerate(models, start=1):
        runtime = runtime_by_model[model_id]
        required_disk_gb = max(
            args.min_free_disk_gb,
            runtime["minimum_free_disk_gb"],
        )
        free = _free_gb(args.cache_root.parent)
        if free < required_disk_gb:
            raise RuntimeError(
                f"solo {free:.1f} GB liberi prima di {model_id}; "
                f"richiesti {required_disk_gb:.1f} GB"
            )
        precision = runtime["quantization"] or runtime["dtype"]
        print(
            f"\n######## [{index}/{len(models)}] {model_id} [{precision}] ########",
            flush=True,
        )
        model_cache, env = _model_environment(args.cache_root, model_id)
        try:
            override = args.models is not None
            if "controlled" in phases and (override or model_id in panels["controlled"]):
                _run_controlled(
                    model_id,
                    env,
                    force=args.force,
                    dtype=runtime["dtype"],
                    quantization=runtime["quantization"],
                )
            if "real" in phases and (override or model_id in panels["real"]):
                _run_real(
                    model_id,
                    args.xlsx,
                    env,
                    force=args.force,
                    dtype=runtime["dtype"],
                    quantization=runtime["quantization"],
                )
            if "reasoning" in phases and (override or model_id in panels["reasoning"]):
                _run_reasoning(
                    model_id,
                    args.xlsx,
                    env,
                    force=args.force,
                    reasoning_max_new_tokens=args.reasoning_max_new_tokens,
                    native_reasoning_max_new_tokens=(
                        args.native_reasoning_max_new_tokens
                    ),
                    dtype=runtime["dtype"],
                    quantization=runtime["quantization"],
                )
        finally:
            if not args.keep_model_cache:
                _remove_model_cache(model_cache, args.cache_root)
                print(f"  ~ cache pesi rimossa: {model_id}", flush=True)

    if args.models:
        print("Pilot/override completato; analisi aggregate intenzionalmente saltate.")
        return 0

    (_ROOT / "outputs/packages").mkdir(parents=True, exist_ok=True)
    _run_aggregate_analysis(phases, args.cohort)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        commit = None
    manifest = {
        "schema_version": "oncoemotion_complete_colab_run/1.0",
        "completed": True,
        "git_commit": commit,
        "phases": phases,
        "cohort": args.cohort,
        "models": models,
        "runtime_profile_id": runtime_config.get("profile_id"),
        "runtime_config_sha256": _sha256(args.runtime_config),
        "model_runtime": runtime_by_model,
        "source_workbook_sha256": _sha256(args.xlsx),
        "reasoning_max_new_tokens": args.reasoning_max_new_tokens,
        "native_reasoning_max_new_tokens": args.native_reasoning_max_new_tokens,
        "model_panels": panels,
        "elapsed_minutes": round((time.time() - started) / 60, 2),
        "privacy": (
            "outputs/tables/item_audit contains raw clinical text and must remain "
            "in the private run directory; public ZIP packages are redaction-checked"
        ),
    }
    manifest_path = _ROOT / "outputs/full_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nRun completo terminato -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
