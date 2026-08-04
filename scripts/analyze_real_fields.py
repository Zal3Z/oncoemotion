#!/usr/bin/env python
"""Confirmatory analysis for clinician-validated real oncology free text.

This analysis is intentionally separate from ``analyze_results.py``: real records
have no neutral/emotional counterfactual pair. It evaluates gold-code performance,
generative abstention, independently learned affect projections, within-item grade
slopes and role-conditioned changes. Identical text strings are retained as distinct
validated assessments and resampled as ``source_id`` clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_rows(path: Path, model: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["model"] = model
            rows.append(row)
    return rows


def _configured_models(config: dict, cohort: str) -> list[str]:
    models = config.get("models", {})
    primary = list(models.get("primary", []))
    secondary = list(models.get("secondary", []))
    return primary if cohort == "primary" else [*primary, *secondary]


def validate_artifacts(
    paths: Iterable[Path],
    config: dict,
    study_path: Path,
    *,
    cohort: str = "primary",
    allow_partial: bool = False,
) -> tuple[list[dict], dict, dict]:
    expected_ids = _configured_models(config, cohort)
    expected = {_slug(model_id): model_id for model_id in expected_ids}
    by_slug = {_slug(path.name.split("__")[0]): path for path in paths}
    missing = sorted(set(expected) - set(by_slug))
    if missing and not allow_partial:
        raise ValueError("real-study cohort is incomplete; missing: " + ", ".join(missing))
    selected = {slug: by_slug[slug] for slug in expected if slug in by_slug}
    if not selected:
        raise ValueError("no configured real-study model rows were found")

    protocol_id = config["protocol_id"]
    design = config["design"]
    expected_roles = list(design["roles"])
    expected_arms = list(design["arms"])
    expected_items = int(design["expected_records_per_model"])
    expected_rows = int(design["expected_rows_per_model"])
    study_hash = _sha256(study_path)

    rows_all: list[dict] = []
    metas = {}
    dataset_hashes = set()
    record_sets = {}
    checks = {}
    for slug, rows_path in selected.items():
        meta_path = rows_path.with_name(f"{slug}__meta.json")
        if not meta_path.exists():
            raise ValueError(f"metadata missing for {slug}: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = _load_rows(rows_path, slug)
        observed_roles = sorted({row.get("role") for row in rows})
        observed_arms = sorted({row.get("arm") for row in rows})
        keys = [(row.get("record_id"), row.get("role"), row.get("arm")) for row in rows]
        model_checks = {
            "model_id": meta.get("model_id") == expected[slug],
            "protocol_id": meta.get("protocol_id") == protocol_id,
            "study_config_sha256": meta.get("study_config_sha256") == study_hash,
            "rows_sha256": meta.get("rows_sha256") == _sha256(rows_path),
            "roles": observed_roles == sorted(expected_roles),
            "arms": observed_arms == sorted(expected_arms),
            "n_items": meta.get("n_items") == expected_items,
            "n_rows": len(rows) == expected_rows == meta.get("n_rows"),
            "unique_cells": len(keys) == len(set(keys)),
            "text_redacted": bool(meta.get("text_redacted"))
            and all(row.get("text_redacted") for row in rows),
            "no_raw_text_in_rows": all(
                row.get("text") == row.get("source_id") for row in rows
            ),
            "real_framing": all(row.get("framing") == "real" for row in rows),
        }
        failed = sorted(name for name, passed in model_checks.items() if not passed)
        if failed:
            raise ValueError(f"artifact validation failed for {slug}: {failed}")
        dataset_hashes.add(meta.get("dataset_sha256"))
        record_sets[slug] = {row["record_id"] for row in rows}
        checks[slug] = model_checks
        metas[slug] = meta
        rows_all.extend(rows)

    if len(dataset_hashes) != 1:
        raise ValueError("models were not run on one common real dataset hash")
    reference_records = next(iter(record_sets.values()))
    mismatched = sorted(slug for slug, records in record_sets.items() if records != reference_records)
    if mismatched:
        raise ValueError("models do not share the same real record set: " + ", ".join(mismatched))

    reference_model = next(iter(selected))
    reference_role = expected_roles[0]
    reference_rows = [
        row for row in rows_all
        if row["model"] == reference_model and row["role"] == reference_role
    ]
    category_counts = Counter(row["category"] for row in reference_rows)
    expected_source = config.get("source_expectations", {})
    observed_source = {
        "records": len(reference_rows),
        "term_records": sum(row["gold_class"] == "term" for row in reference_rows),
        "no_direct_pro_match_records": category_counts.get("NO_DIRECT_PRO_MATCH", 0),
        "insufficient_context_records": category_counts.get("INSUFFICIENT_CONTEXT", 0),
        "distinct_pro_ids": len({row["gold_pro_id"] for row in reference_rows if row["gold_pro_id"]}),
        "unique_source_ids": len({row["source_id"] for row in reference_rows}),
        "records_ge_7_words": sum(int(row.get("n_words") or 0) >= 7 for row in reference_rows),
    }
    source_failures = {
        key: {"expected": value, "observed": observed_source.get(key)}
        for key, value in expected_source.items()
        if key in observed_source and observed_source.get(key) != value
    }
    if source_failures:
        raise ValueError(f"real source expectations failed: {source_failures}")

    return rows_all, metas, {
        "passed": True,
        "cohort": cohort,
        "models": sorted(selected),
        "missing_allowed": missing,
        "checks_by_model": checks,
        "dataset_sha256": next(iter(dataset_hashes)),
        "source_counts": observed_source,
    }


def hierarchical_cluster_stat_ci(
    rows: list[dict],
    statistic: Callable[[list[dict]], float],
    *,
    n_boot: int,
    seed: int,
    cluster_field: str = "source_id",
) -> dict:
    """Equal-model bootstrap with source-text clusters resampled within model."""
    by_model: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cluster = str(row.get(cluster_field) or row.get("record_id"))
        by_model[str(row["model"])][cluster].append(row)
    models = sorted(by_model)
    if not models:
        return {"estimate": None, "ci95": [None, None], "n_models": 0, "n_rows": 0}

    model_points = [statistic([row for group in by_model[m].values() for row in group]) for m in models]
    model_points = [value for value in model_points if _finite(value)]
    point = float(np.mean(model_points)) if model_points else float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        model_values = []
        for model in sampled_models:
            groups = by_model[str(model)]
            cluster_ids = np.array(sorted(groups), dtype=object)
            sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sampled_rows = [row for cluster in sampled_clusters for row in groups[str(cluster)]]
            value = statistic(sampled_rows)
            if _finite(value):
                model_values.append(float(value))
        if model_values:
            boots.append(float(np.mean(model_values)))
    if not boots or not _finite(point):
        interval = [None, None]
        estimate = None
    else:
        lo, hi = np.quantile(np.asarray(boots), [0.025, 0.975])
        interval = [round(float(lo), 6), round(float(hi), 6)]
        estimate = round(point, 6)
    return {
        "estimate": estimate,
        "ci95": interval,
        "n_models": len(models),
        "n_rows": len(rows),
        "n_source_clusters": len({(row["model"], row.get(cluster_field)) for row in rows}),
    }


def _accuracy(rows: list[dict]) -> float:
    values = [float(bool(row.get("correct"))) for row in rows if row.get("correct") is not None]
    return float(np.mean(values)) if values else float("nan")


def _macro_recall(rows: list[dict]) -> float:
    by_gold: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("gold_pro_id") and row.get("correct") is not None:
            by_gold[row["gold_pro_id"]].append(float(bool(row["correct"])))
    return float(np.mean([np.mean(values) for values in by_gold.values()])) if by_gold else float("nan")


def _abstention_rate(rows: list[dict]) -> float:
    values = [float(row.get("generative_kind") == "abstained") for row in rows]
    return float(np.mean(values)) if values else float("nan")


def _false_positive_rate(rows: list[dict]) -> float:
    values = [float(row.get("generative_top1_id") is not None) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def _z_value(row: dict, readout: str, concept: str, *, control: bool = False):
    field = readout
    if control:
        field = "z_controls_read" if readout == "z_read" else "z_controls"
    values = row.get(field) or {}
    value = values.get(concept)
    return float(value) if _finite(value) else None


def _add_affect_composite(rows: list[dict], readout: str, axes: list[str]) -> None:
    for row in rows:
        values = [_z_value(row, readout, axis) for axis in axes]
        row["_affect_composite"] = (
            float(np.mean(values)) if values and all(value is not None for value in values) else None
        )


def _center_by_group(values: list[float], groups: list[tuple]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    centered = np.empty_like(array)
    positions: dict[tuple, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        positions[group].append(index)
    for indices in positions.values():
        centered[indices] = array[indices] - float(np.mean(array[indices]))
    return centered


def _residual_association(
    rows: list[dict],
    exposure: Callable[[dict], float | None],
    outcome: Callable[[dict], float | None],
    group: Callable[[dict], tuple],
    nuisances: list[Callable[[dict], float | None]] | None = None,
) -> float:
    usable = []
    for row in rows:
        x, y = exposure(row), outcome(row)
        nuisance_values = [fn(row) for fn in (nuisances or [])]
        if x is None or y is None or not _finite(x) or not _finite(y):
            continue
        if any(value is None or not _finite(value) for value in nuisance_values):
            continue
        usable.append((row, float(x), float(y), [float(value) for value in nuisance_values]))
    if len(usable) < 10:
        return float("nan")
    groups = [group(row) for row, *_ in usable]
    x = _center_by_group([item[1] for item in usable], groups)
    y = _center_by_group([item[2] for item in usable], groups)
    if nuisances:
        columns = []
        for nuisance_index in range(len(nuisances)):
            columns.append(_center_by_group(
                [item[3][nuisance_index] for item in usable], groups))
        nuisance_matrix = np.column_stack(columns)
        keep = np.std(nuisance_matrix, axis=0) > 1e-12
        if keep.any():
            nuisance_matrix = nuisance_matrix[:, keep]
            x = x - nuisance_matrix @ np.linalg.lstsq(nuisance_matrix, x, rcond=None)[0]
            y = y - nuisance_matrix @ np.linalg.lstsq(nuisance_matrix, y, rcond=None)[0]
    sd = float(np.std(x))
    if sd < 1e-12:
        return float("nan")
    x = x / sd
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def _grade_slope(rows: list[dict], value: Callable[[dict], float | None]) -> float:
    return _residual_association(
        rows,
        exposure=lambda row: float(row["grade"]) if row.get("grade") is not None else None,
        outcome=value,
        group=lambda row: (row.get("source_item"),),
        nuisances=[lambda row: math.log1p(float(row.get("n_words") or 0))],
    )


def _error_affect_slope(rows: list[dict]) -> float:
    return _residual_association(
        rows,
        exposure=lambda row: row.get("_affect_composite"),
        outcome=lambda row: float(not bool(row["correct"]))
        if row.get("correct") is not None else None,
        group=lambda row: (row.get("source_item"), row.get("grade")),
        nuisances=[
            lambda row: math.log1p(float(row.get("n_words") or 0)),
            lambda row: _z_value(row, "z_read", "clinical_severity", control=True),
            lambda row: _z_value(row, "z_read", "general_negative_valence", control=True),
        ],
    )


def _paired_role_rows(rows: list[dict], role: str, control: str, axes: list[str]) -> list[dict]:
    index: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("arm") == "intact" and row.get("role") in {role, control}:
            index[(row["model"], row["record_id"])][row["role"]] = row
    paired = []
    for (model, record_id), cells in index.items():
        if role not in cells or control not in cells:
            continue
        treated, reference = cells[role], cells[control]
        derived = {
            "model": model,
            "record_id": record_id,
            "source_id": treated.get("source_id"),
            "gold_class": treated.get("gold_class"),
            "n_words": treated.get("n_words"),
            "label_disagreement": float(
                treated.get("model_top1_id") != reference.get("model_top1_id")
            ),
        }
        for axis in axes:
            a = _z_value(treated, "z_read", axis)
            b = _z_value(reference, "z_read", axis)
            derived[f"delta_{axis}"] = a - b if a is not None and b is not None else None
        a = treated.get("_affect_composite")
        b = reference.get("_affect_composite")
        derived["delta_affect_composite"] = a - b if a is not None and b is not None else None
        paired.append(derived)
    return paired


def _mean_field(field: str) -> Callable[[list[dict]], float]:
    def statistic(rows: list[dict]) -> float:
        values = [float(row[field]) for row in rows if _finite(row.get(field))]
        return float(np.mean(values)) if values else float("nan")
    return statistic


def _per_model_role(rows: list[dict], roles: list[str], axes: list[str]) -> dict:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        result[model] = {}
        for role in roles:
            selected = [row for row in rows if row["model"] == model and row["role"] == role]
            term = [row for row in selected if row["gold_class"] == "term"]
            abstain = [row for row in selected if row["gold_class"] == "abstain"]
            axis_summary = {}
            for axis in axes:
                values = [_z_value(row, "z_read", axis) for row in selected]
                values = [value for value in values if value is not None]
                if values:
                    axis_summary[axis] = {
                        "mean_z_read": round(float(np.mean(values)), 5),
                        "fraction_above_neutral_mean": round(float(np.mean(np.asarray(values) > 0)), 5),
                    }
            result[model][role] = {
                "n_records": len(selected),
                "term_accuracy": round(_accuracy(term), 5),
                "term_macro_recall": round(_macro_recall(term), 5),
                "abstention_rate_non_pro": round(_abstention_rate(abstain), 5),
                "false_positive_rate_non_pro": round(_false_positive_rate(abstain), 5),
                "affect": axis_summary,
            }
    return result


def _unambiguous_source_ids(rows: list[dict], model: str, role: str) -> set[str]:
    signatures: dict[str, set[tuple]] = defaultdict(set)
    for row in rows:
        if row["model"] != model or row["role"] != role:
            continue
        signatures[row["source_id"]].add((
            row.get("annotation_source"), row.get("gold_pro_id"),
            row.get("source_item"), row.get("grade"),
        ))
    return {source_id for source_id, values in signatures.items() if len(values) == 1}


def analyze(rows: list[dict], metas: dict, config: dict, *, n_boot: int, seed: int) -> dict:
    roles = list(config["design"]["roles"])
    role = config["primary"]["role"]
    control = config["primary"]["control"]
    axes = list(config["affective_profile"]["concepts"])
    primary_axes = list(config["primary"]["affect_composite"])
    eligible = {model: set(meta.get("eligible_affect_axes") or []) for model, meta in metas.items()}
    ineligible_primary = {
        model: sorted(set(primary_axes) - available)
        for model, available in eligible.items()
        if not set(primary_axes).issubset(available)
    }
    _add_affect_composite(rows, config["design"]["readout_primary"], primary_axes)

    term_role = [
        row for row in rows
        if row["role"] == role and row["arm"] == "intact" and row["gold_class"] == "term"
    ]
    non_pro_role = [
        row for row in rows
        if row["role"] == role and row["arm"] == "intact" and row["gold_class"] == "abstain"
    ]
    paired = _paired_role_rows(rows, role, control, primary_axes)
    paired_term = [row for row in paired if row.get("gold_class") == "term"]

    pooled = {
        "term_accuracy": hierarchical_cluster_stat_ci(
            term_role, _accuracy, n_boot=n_boot, seed=seed),
        "term_macro_recall": hierarchical_cluster_stat_ci(
            term_role, _macro_recall, n_boot=n_boot, seed=seed + 1),
        "non_pro_abstention_rate": hierarchical_cluster_stat_ci(
            non_pro_role, _abstention_rate, n_boot=n_boot, seed=seed + 2),
        "non_pro_false_positive_rate": hierarchical_cluster_stat_ci(
            non_pro_role, _false_positive_rate, n_boot=n_boot, seed=seed + 3),
        "role_top1_disagreement": hierarchical_cluster_stat_ci(
            paired_term, _mean_field("label_disagreement"), n_boot=n_boot, seed=seed + 4),
    }

    affect = {
        "primary_axes": primary_axes,
        "primary_gate_passes_all_models": not ineligible_primary,
        "ineligible_primary_axes_by_model": ineligible_primary,
        "eligible_axes_by_model": {model: sorted(values) for model, values in eligible.items()},
        "interpretation": (
            "Projection means are standardized against the neutral baseline and do not imply "
            "that the model experiences emotion. Real-text associations are observational."
        ),
    }
    if not ineligible_primary:
        affect["primary_error_association"] = hierarchical_cluster_stat_ci(
            term_role, _error_affect_slope, n_boot=n_boot, seed=seed + 10)
        affect["primary_error_association"]["estimand"] = (
            "absolute error-probability change per within-model SD of the fear/concern "
            "composite, after centering within source item and grade and adjusting for "
            "text length, clinical severity and general negative valence"
        )
        affect["within_item_grade_slope"] = hierarchical_cluster_stat_ci(
            term_role,
            lambda selected: _grade_slope(selected, lambda row: row.get("_affect_composite")),
            n_boot=n_boot,
            seed=seed + 11,
        )
        affect["role_shift_affect_composite"] = hierarchical_cluster_stat_ci(
            paired_term,
            _mean_field("delta_affect_composite"),
            n_boot=n_boot,
            seed=seed + 12,
        )

    grade_slopes = {}
    for model in sorted(metas):
        grade_slopes[model] = {}
        for which_role in roles:
            selected = [
                row for row in rows
                if row["model"] == model and row["role"] == which_role
                and row["arm"] == "intact" and row["gold_class"] == "term"
            ]
            model_result = {}
            for axis in axes:
                if axis in eligible.get(model, set()):
                    value = _grade_slope(selected, lambda row, name=axis: _z_value(row, "z_read", name))
                    model_result[axis] = round(value, 6) if _finite(value) else None
            for control_axis in ("clinical_severity", "general_negative_valence"):
                value = _grade_slope(
                    selected,
                    lambda row, name=control_axis: _z_value(row, "z_read", name, control=True),
                )
                model_result[f"control:{control_axis}"] = round(value, 6) if _finite(value) else None
            grade_slopes[model][which_role] = model_result
    affect["within_item_grade_slopes_per_model_role"] = grade_slopes

    first_model = sorted(metas)[0]
    unambiguous = _unambiguous_source_ids(rows, first_model, role)
    sensitivity_rows = [row for row in term_role if row.get("source_id") in unambiguous]
    long_rows = [row for row in term_role if int(row.get("n_words") or 0) >= 7]
    sensitivity = {
        "unambiguous_identical_text_clusters": {
            "n_source_ids": len(unambiguous),
            "term_accuracy": hierarchical_cluster_stat_ci(
                sensitivity_rows, _accuracy, n_boot=n_boot, seed=seed + 20),
        },
        "texts_ge_7_words": {
            "n_rows": len(long_rows),
            "term_accuracy": hierarchical_cluster_stat_ci(
                long_rows, _accuracy, n_boot=n_boot, seed=seed + 21),
        },
    }
    if not ineligible_primary:
        sensitivity["unambiguous_identical_text_clusters"]["error_association"] = (
            hierarchical_cluster_stat_ci(
                sensitivity_rows, _error_affect_slope, n_boot=n_boot, seed=seed + 22
            )
        )
        sensitivity["texts_ge_7_words"]["error_association"] = hierarchical_cluster_stat_ci(
            long_rows, _error_affect_slope, n_boot=n_boot, seed=seed + 23
        )

    return {
        "models": sorted(metas),
        "roles": roles,
        "n_rows": len(rows),
        "per_model_role": _per_model_role(rows, roles, axes),
        "pooled_primary_role": pooled,
        "affect": affect,
        "sensitivity": sensitivity,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="*", type=Path)
    ap.add_argument(
        "--study-config",
        type=Path,
        default=_ROOT / "configs/study_esmo_2026_real.yaml",
    )
    ap.add_argument("--cohort", choices=["primary", "all"], default="primary")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "outputs/real_fields/real_primary_analysis.json",
    )
    ap.add_argument("--bootstrap-draws", type=int, default=0)
    args = ap.parse_args()

    paths = args.rows or sorted((_ROOT / "outputs/real_fields").glob("*__rows.jsonl"))
    config = yaml.safe_load(args.study_config.read_text(encoding="utf-8")) or {}
    rows, metas, validation = validate_artifacts(
        paths,
        config,
        args.study_config,
        cohort=args.cohort,
        allow_partial=args.allow_partial,
    )
    analysis_cfg = config.get("analysis", {})
    n_boot = args.bootstrap_draws or int(analysis_cfg.get("bootstrap_draws", 5000))
    seed = int(analysis_cfg.get("seed", 20261001))
    report = {
        "protocol_id": config.get("protocol_id"),
        "study_config_sha256": _sha256(args.study_config),
        "artifact_validation": validation,
        "analysis": analyze(rows, metas, config, n_boot=n_boot, seed=seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = report["analysis"]["affect"].get("primary_error_association")
    print(f"{len(rows)} rows | {len(metas)} models | protocol={config.get('protocol_id')}")
    print(f"term accuracy: {report['analysis']['pooled_primary_role']['term_accuracy']}")
    print(f"primary affect/error association: {primary or 'unavailable (axis gate)'}")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
