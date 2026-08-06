#!/usr/bin/env python
"""Analyse the explicit-abstention direct/deliberative oncology extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path, model: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["model"] = model
            rows.append(row)
    return rows


def _configured_models(config: dict, cohort: str) -> list[str]:
    primary = list(config.get("models", {}).get("primary", []))
    secondary = list(config.get("models", {}).get("secondary", []))
    return primary if cohort == "primary" else [*primary, *secondary]


def _reasoning_modes_for_model(config: dict, model_id: str) -> list[str]:
    modes = list(config["design"]["reasoning_modes"])
    native = config.get("native_reasoning", {}) or {}
    if native.get("enabled") and model_id in set(native.get("models", [])):
        modes.append(str(native.get("mode", "native_reasoning")))
    return list(dict.fromkeys(modes))


def validate_artifacts(
    paths: list[Path],
    config: dict,
    study_path: Path,
    *,
    cohort: str,
    allow_partial: bool = False,
) -> tuple[list[dict], dict]:
    expected_ids = _configured_models(config, cohort)
    expected = {_slug(model_id): model_id for model_id in expected_ids}
    supplied = {_slug(path.name.split("__")[0]): path for path in paths}
    missing = sorted(set(expected) - set(supplied))
    if missing and not allow_partial:
        raise ValueError("reasoning cohort is incomplete; missing: " + ", ".join(missing))
    selected = {slug: supplied[slug] for slug in expected if slug in supplied}
    if not selected:
        raise ValueError("no configured reasoning-study rows were found")

    design = config["design"]
    roles = list(design["roles"])
    base_modes = list(design["reasoning_modes"])
    expected_items = int(design["expected_records_per_model"])
    study_hash = _sha256(study_path)
    all_rows: list[dict] = []
    checks: dict[str, dict] = {}
    datasets = set()
    record_sets = {}
    for slug, path in selected.items():
        meta_path = path.with_name(f"{slug}__meta.json")
        if not meta_path.exists():
            raise ValueError(f"metadata missing for {slug}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = _load_rows(path, slug)
        modes = _reasoning_modes_for_model(config, expected[slug])
        expected_rows = expected_items * len(roles) * len(modes)
        keys = [
            (row.get("record_id"), row.get("role"), row.get("reasoning_mode"))
            for row in rows
        ]
        expected_backends = {
            "direct": "none",
            "deliberative": "standardized_prompted",
            "native_reasoning": "native_chat_template_thinking",
        }
        model_checks = {
            "model_id": meta.get("model_id") == expected[slug],
            "protocol_id": meta.get("protocol_id") == config.get("protocol_id"),
            "study_config_sha256": meta.get("study_config_sha256") == study_hash,
            "dataset_sha256": bool(meta.get("dataset_sha256")),
            "rows_sha256": meta.get("rows_sha256") == _sha256(path),
            "roles": sorted({row.get("role") for row in rows}) == sorted(roles),
            "reasoning_modes": (
                sorted({row.get("reasoning_mode") for row in rows}) == sorted(modes)
            ),
            "decision_space": all(row.get("decision_space") == "joint" for row in rows),
            "n_items": meta.get("n_items") == expected_items,
            "n_rows": len(rows) == expected_rows == meta.get("n_rows"),
            "unique_cells": len(keys) == len(set(keys)),
            "text_redacted": all(
                row.get("text_redacted") is True
                and row.get("text") == row.get("source_id")
                for row in rows
            ),
            "reasoning_redacted": all(row.get("reasoning_text") is None for row in rows),
            "reasoning_backend": all(
                row.get("reasoning_backend")
                == expected_backends.get(row.get("reasoning_mode"))
                for row in rows
            ),
            "native_thinking_scope": all(
                bool(row.get("native_thinking_enabled"))
                == (row.get("reasoning_mode") == "native_reasoning")
                for row in rows
            ),
        }
        failed = sorted(name for name, passed in model_checks.items() if not passed)
        if failed:
            raise ValueError(f"artifact validation failed for {slug}: {failed}")
        checks[slug] = model_checks
        datasets.add(meta["dataset_sha256"])
        record_sets[slug] = {row["record_id"] for row in rows}
        all_rows.extend(rows)
    if len(datasets) != 1:
        raise ValueError("models were not run on one common real dataset hash")
    reference = next(iter(record_sets.values()))
    if any(records != reference for records in record_sets.values()):
        raise ValueError("models do not share the same validated assessment rows")

    ref_model = next(iter(selected))
    ref_rows = [
        row
        for row in all_rows
        if row["model"] == ref_model
        and row["role"] == roles[0]
        and row["reasoning_mode"] == base_modes[0]
    ]
    systems = Counter(row["gold_system"] for row in ref_rows)
    observed = {
        "records": len(ref_rows),
        "pro_records": systems.get("PRO-CTCAE", 0),
        "ctcae_records": systems.get("CTCAE", 0),
        "nonclassifiable_records": systems.get("NON_CLASSIFICABILE", 0),
        "distinct_pro_ids": len({row["gold_pro_id"] for row in ref_rows if row.get("gold_pro_id")}),
        "distinct_ctcae_items": len(
            {row["gold_ctcae_term"] for row in ref_rows if row.get("gold_ctcae_term")}
        ),
        "unique_source_ids": len({row["source_id"] for row in ref_rows}),
    }
    expected_source = config.get("source_expectations", {})
    failures = {
        key: {"expected": expected_source[key], "observed": observed.get(key)}
        for key in expected_source
        if observed.get(key) != expected_source[key]
    }
    if failures:
        raise ValueError(f"reasoning source expectations failed: {failures}")
    return all_rows, {
        "passed": True,
        "cohort": cohort,
        "models": sorted(selected),
        "missing_allowed": missing,
        "checks_by_model": checks,
        "dataset_sha256": next(iter(datasets)),
        "source_counts": observed,
    }


def _rate(rows: list[dict], field: str) -> float | None:
    values = [float(bool(row[field])) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _metrics(rows: list[dict]) -> dict:
    covered = [row for row in rows if not row.get("explicit_nonclassifiable")]
    return {
        "n_rows": len(rows),
        "strict_joint_accuracy": _rate(rows, "correct"),
        "ontology_accuracy": _rate(rows, "system_correct"),
        "pro_item_accuracy": _rate(rows, "pro_correct"),
        "ctcae_item_accuracy": _rate(rows, "ctcae_correct"),
        "nonclassifiable_accuracy": _rate(rows, "nonclassifiable_correct"),
        "explicit_nonclassifiable_rate": _rate(rows, "explicit_nonclassifiable"),
        "coverage": len(covered) / len(rows) if rows else None,
        "selective_accuracy": _rate(covered, "correct"),
    }


def _hierarchical_ci(
    rows: list[dict],
    value,
    *,
    n_boot: int,
    seed: int,
) -> dict:
    by_model: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model[str(row["model"])][str(row.get("source_id") or row["record_id"])].append(row)
    models = sorted(by_model)

    def model_stat(selected: list[dict]) -> float:
        values = [float(v) for row in selected if (v := value(row)) is not None and math.isfinite(float(v))]
        return float(np.mean(values)) if values else float("nan")

    points = []
    for model in models:
        selected = [row for group in by_model[model].values() for row in group]
        stat = model_stat(selected)
        if math.isfinite(stat):
            points.append(stat)
    estimate = float(np.mean(points)) if points else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        model_values = []
        for model in sampled_models:
            groups = by_model[str(model)]
            source_ids = np.asarray(sorted(groups), dtype=object)
            sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
            stat = model_stat([row for source_id in sampled for row in groups[str(source_id)]])
            if math.isfinite(stat):
                model_values.append(stat)
        if model_values:
            boots.append(float(np.mean(model_values)))
    if not boots or not math.isfinite(estimate):
        return {"estimate": None, "ci95": [None, None], "n_models": len(models)}
    lo, hi = np.quantile(np.asarray(boots), [0.025, 0.975])
    return {
        "estimate": round(estimate, 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "n_models": len(models),
        "n_rows": len(rows),
    }


def _paired_mode_rows(
    rows: list[dict],
    role: str,
    *,
    comparison_mode: str,
    baseline_mode: str = "direct",
) -> list[dict]:
    cells: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["role"] == role:
            cells[(row["model"], row["record_id"])][row["reasoning_mode"]] = row
    paired = []
    for (model, record_id), modes in cells.items():
        if {baseline_mode, comparison_mode}.issubset(modes):
            baseline = modes[baseline_mode]
            comparison = modes[comparison_mode]
            paired.append(
                {
                    "model": model,
                    "record_id": record_id,
                    "source_id": baseline["source_id"],
                    "accuracy_delta": float(comparison["correct"])
                    - float(baseline["correct"]),
                    "flipped": comparison["model_choice_id"]
                    != baseline["model_choice_id"],
                }
            )
    return paired


def analyse(rows: list[dict], config: dict) -> dict:
    design = config["design"]
    n_boot = int(config.get("analysis", {}).get("bootstrap_draws", 5000))
    seed = int(config.get("analysis", {}).get("seed", 20261002))
    primary_role = str(config.get("analysis", {}).get("primary_role", "oncologo"))

    per_model: dict[str, dict] = defaultdict(dict)
    configured_by_slug = {
        _slug(model_id): model_id
        for model_id in _configured_models(config, "all")
    }
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for role in design["roles"]:
            per_model[model][role] = {}
            modes = _reasoning_modes_for_model(config, configured_by_slug[model])
            for mode in modes:
                selected = [
                    row
                    for row in model_rows
                    if row["role"] == role and row["reasoning_mode"] == mode
                ]
                per_model[model][role][mode] = _metrics(selected)

    primary_rows = [row for row in rows if row["role"] == primary_role]
    pooled = {}
    for mode_index, mode in enumerate(design["reasoning_modes"]):
        selected = [row for row in primary_rows if row["reasoning_mode"] == mode]
        pooled[mode] = {
            "strict_joint_accuracy": _hierarchical_ci(
                selected,
                lambda row: float(row["correct"]),
                n_boot=n_boot,
                seed=seed + mode_index,
            ),
            "ontology_accuracy": _hierarchical_ci(
                selected,
                lambda row: float(row["system_correct"]),
                n_boot=n_boot,
                seed=seed + 10 + mode_index,
            ),
            "explicit_nonclassifiable_rate": _hierarchical_ci(
                selected,
                lambda row: float(row["explicit_nonclassifiable"]),
                n_boot=n_boot,
                seed=seed + 20 + mode_index,
            ),
        }

    paired = _paired_mode_rows(
        rows,
        primary_role,
        comparison_mode="deliberative",
    )
    reasoning_effect = {
        "strict_accuracy_delta_deliberative_minus_direct": _hierarchical_ci(
            paired,
            lambda row: row["accuracy_delta"],
            n_boot=n_boot,
            seed=seed + 100,
        ),
        "choice_flip_rate": _hierarchical_ci(
            paired,
            lambda row: float(row["flipped"]),
            n_boot=n_boot,
            seed=seed + 101,
        ),
    }

    native_results = {}
    native_config = config.get("native_reasoning", {}) or {}
    native_mode = str(native_config.get("mode", "native_reasoning"))
    for model_id in native_config.get("models", []):
        model = _slug(model_id)
        if model not in per_model:
            continue
        native_rows = [
            row
            for row in primary_rows
            if row["model"] == model and row["reasoning_mode"] == native_mode
        ]
        paired_native = [
            row
            for row in _paired_mode_rows(
                rows,
                primary_role,
                comparison_mode=native_mode,
            )
            if row["model"] == model
        ]
        native_results[model] = {
            "model_id": model_id,
            "status": "single-model exploratory; not pooled with standardized deliberation",
            "metrics": _metrics(native_rows),
            "strict_accuracy_delta_native_minus_direct": _hierarchical_ci(
                paired_native,
                lambda row: row["accuracy_delta"],
                n_boot=n_boot,
                seed=seed + 200,
            ),
            "choice_flip_rate_native_vs_direct": _hierarchical_ci(
                paired_native,
                lambda row: float(row["flipped"]),
                n_boot=n_boot,
                seed=seed + 201,
            ),
        }

    def paired_summaries(tier: str) -> list[dict]:
        summaries = []
        for pair in config.get("model_pairs", {}).get(tier, []):
            base = _slug(pair["base"])
            med = _slug(pair["medicalized"])
            if base not in per_model or med not in per_model:
                continue
            direct_base = per_model[base][primary_role]["direct"]["strict_joint_accuracy"]
            direct_med = per_model[med][primary_role]["direct"]["strict_joint_accuracy"]
            delib_base = per_model[base][primary_role]["deliberative"][
                "strict_joint_accuracy"
            ]
            delib_med = per_model[med][primary_role]["deliberative"][
                "strict_joint_accuracy"
            ]
            summaries.append(
                {
                    "family": pair["family"],
                    "analysis_tier": tier,
                    "base_slug": base,
                    "medicalized_slug": med,
                    "direct": {
                        "base": direct_base,
                        "medicalized": direct_med,
                        "medicalized_minus_base": direct_med - direct_base,
                    },
                    "deliberative": {
                        "base": delib_base,
                        "medicalized": delib_med,
                        "medicalized_minus_base": delib_med - delib_base,
                    },
                    "medicalization_x_deliberation_interaction": (
                        (delib_med - delib_base) - (direct_med - direct_base)
                    ),
                }
            )
        return summaries

    paired_families = paired_summaries("primary")
    paired_medical_references = paired_summaries("secondary")
    return {
        "primary_role": primary_role,
        "per_model_role_reasoning": per_model,
        "pooled_primary_role": pooled,
        "reasoning_effect": reasoning_effect,
        "native_reasoning_exploratory": native_results,
        "paired_base_medicalized": paired_families,
        "paired_medical_references": paired_medical_references,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument(
        "--study-config",
        type=Path,
        default=_ROOT / "configs/study_esmo_2026_reasoning.yaml",
    )
    ap.add_argument("--cohort", choices=["primary", "all"], default="primary")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs/reasoning_real/reasoning_analysis.json",
    )
    args = ap.parse_args()
    config = yaml.safe_load(args.study_config.read_text(encoding="utf-8")) or {}
    rows, validation = validate_artifacts(
        args.rows,
        config,
        args.study_config,
        cohort=args.cohort,
        allow_partial=args.allow_partial,
    )
    report = {
        "protocol_id": config.get("protocol_id"),
        "study_config_sha256": _sha256(args.study_config),
        "artifact_validation": validation,
        "analysis": analyse(rows, config),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
