#!/usr/bin/env python
"""Definitive pooled analysis for the ESMO AI 2026 role-by-affect study.

The primary endpoint is the within-item disagreement rate between the oncologist
persona and a token-matched, identity-free filler control.  Accuracy is deliberately
secondary: the abstract's safety claim is that aggregate accuracy can remain
equivalent while the individual PRO-CTCAE codes change.

Inference uses an equal-model hierarchical bootstrap.  Clinical pairs (neutral and
emotional formulations of one seed) stay clustered, and models are resampled at the
outer level.  This is intentionally more conservative than treating every generated
row as an independent observation.

The key secondary analysis separates preclassified patient-affective qualifiers
from symptom-intensity qualifiers, then estimates whether affective wording modifies
the cross-role code-disagreement rate. Internal affect readouts and matched
base/medicalized comparisons remain secondary to observable coding behavior.

Usage:
    python scripts/analyze_results.py \
      --rows-glob "outputs/role_emotion/*__rows.jsonl"
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from oncoemotion.statistics import hierarchical_cluster_ci


def _load(paths) -> list[dict]:
    rows = []
    for path in paths:
        model = Path(path).name.split("__")[0]
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["model"] = model
            row.setdefault("arm", "emotion" if row.get("ablated") else "intact")
            rows.append(row)
    return rows


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_tier_paths(paths, config, study_config):
    """Select the frozen Tier 1 and validate every rows/metadata pair."""
    expected = {_slug(model_id): model_id for model_id in config["models"]["tier1"]}
    by_slug = defaultdict(list)
    for path in map(Path, paths):
        by_slug[path.name.split("__")[0]].append(path)

    missing = sorted(set(expected) - set(by_slug))
    if missing:
        raise ValueError(
            "definitive Tier 1 is incomplete; missing model outputs: " + ", ".join(missing)
        )
    repeated = {slug: items for slug, items in by_slug.items()
                if slug in expected and len(items) != 1}
    if repeated:
        raise ValueError(f"multiple rows files matched the same Tier 1 model: {sorted(repeated)}")

    selected = [by_slug[slug][0] for slug in sorted(expected)]
    excluded = sorted(set(by_slug) - set(expected))
    design = config["design"]
    expected_roles = design["roles"]
    expected_arms = design["arms"]
    expected_items = int(design["expected_total_records_per_model"])
    config_hash = _sha256(study_config)
    metadata = {}
    errors = []
    for path in selected:
        slug = path.name.split("__")[0]
        meta_path = path.with_name(f"{slug}__meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{slug}: missing/invalid metadata ({exc})")
            continue
        checks = {
            "model_id": meta.get("model_id") == expected[slug],
            "protocol_id": meta.get("protocol_id") == config["protocol_id"],
            "study_config_sha256": meta.get("study_config_sha256") == config_hash,
            "roles": meta.get("roles") == expected_roles,
            "arms_requested": meta.get("arms_requested") == expected_arms,
            "scorer": meta.get("scorer") == "both",
            "n_items": meta.get("n_items") == expected_items,
            "rows_sha256": meta.get("rows_sha256") == _sha256(path),
            "ablate_concepts_requested": meta.get("ablate_concepts_requested")
            == config.get("mechanistic_gate", {}).get("target_axes", []),
            "eligible_affect_axes": isinstance(meta.get("eligible_affect_axes"), list),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{slug}: metadata mismatch in {', '.join(failed)}")
        metadata[slug] = meta
    if errors:
        raise ValueError("artifact validation failed:\n  - " + "\n  - ".join(errors))

    dataset_hashes = {meta.get("dataset_sha256") for meta in metadata.values()}
    commits = {meta.get("git_commit") for meta in metadata.values()}
    if len(dataset_hashes) != 1 or None in dataset_hashes:
        raise ValueError("Tier 1 models were not run on one common dataset hash")
    if len(commits) != 1 or None in commits:
        raise ValueError("Tier 1 models were not run from one common git commit")

    return selected, excluded, {
        "passed": True,
        "metadata_files": len(metadata),
        "rows_hashes_match": True,
        "study_config_hash": config_hash,
        "dataset_hash": next(iter(dataset_hashes)),
        "git_commit": next(iter(commits)),
        "eligible_affect_axes_by_model": {
            slug: meta.get("eligible_affect_axes", [])
            for slug, meta in sorted(metadata.items())
        },
    }


def _label(row):
    """Committed 80-way code, with a compatibility path for legacy free runs."""
    if row.get("scorer") == "constrained" or "label_margin" in row:
        return row.get("model_top1_id")
    if "rescored_top1_id" in row:
        return row.get("rescored_top1_id")
    return row.get("model_top1_id")


def _generative_kind(row):
    """Free-generation outcome; never infer abstention from constrained scoring."""
    if row.get("generative_kind") is not None:
        return row["generative_kind"]
    if row.get("scorer") == "constrained":
        return None
    if row.get("rescored_kind") is not None:
        return row["rescored_kind"]
    if row.get("abstained") is True:
        return "abstained"
    if row.get("non_answer") is True:
        return "non_answer"
    if row.get("unmappable") is True:
        return "unmapped"
    if row.get("model_matched") is True:
        return "mapped"
    return None


def _nested_add(store, model, pair_id, value):
    store[model][pair_id].append(float(value))


def _validate_primary_rows(rows, role, control, *, expected_records=None):
    """Reject partial/duplicated definitive results before any inference."""
    seen = Counter(
        (row["model"], row.get("record_id"), row.get("role"), row.get("arm"))
        for row in rows
    )
    duplicates = [key for key, count in seen.items() if count > 1]
    if duplicates:
        preview = ", ".join(map(str, duplicates[:3]))
        raise ValueError(f"duplicate model/record/role/arm rows detected: {preview}")

    cells = defaultdict(dict)
    for row in rows:
        if row.get("arm") != "intact" or row.get("gold_class") != "term":
            continue
        if row.get("role") in (role, control):
            cells[(row["model"], row["record_id"])][row["role"]] = row

    incomplete = [key for key, value in cells.items() if set(value) != {role, control}]
    if incomplete:
        preview = ", ".join(map(str, incomplete[:3]))
        raise ValueError(f"incomplete primary role pairs detected: {preview}")

    by_model = defaultdict(set)
    invalid_labels = []
    gold_mismatches = []
    for (model, record_id), value in cells.items():
        if not value:
            continue
        by_model[model].add(record_id)
        treated, reference = value[role], value[control]
        if _label(treated) is None or _label(reference) is None:
            invalid_labels.append((model, record_id))
        if treated.get("gold_pro_id") != reference.get("gold_pro_id"):
            gold_mismatches.append((model, record_id))

    if invalid_labels:
        raise ValueError(f"missing committed labels in {len(invalid_labels)} primary pairs")
    if gold_mismatches:
        raise ValueError(f"gold-label mismatch in {len(gold_mismatches)} primary pairs")
    if not by_model:
        raise ValueError("no complete primary term records found")

    counts = {model: len(ids) for model, ids in sorted(by_model.items())}
    if expected_records is not None:
        wrong = {model: count for model, count in counts.items() if count != expected_records}
        if wrong:
            raise ValueError(
                f"partial primary results: expected {expected_records} term records per model; "
                f"observed {wrong}"
            )
    reference_ids = next(iter(by_model.values()))
    nonidentical = [model for model, ids in by_model.items() if ids != reference_ids]
    if nonidentical:
        raise ValueError(
            "models do not contain the same primary record set: " + ", ".join(nonidentical)
        )

    return {
        "passed": True,
        "duplicate_rows": 0,
        "incomplete_role_pairs": 0,
        "models": len(by_model),
        "term_records_per_model": counts,
        "common_record_set": True,
    }


def _validate_affective_subset(
    rows,
    *,
    field,
    target,
    expected_pairs=None,
):
    """Ensure the preregistered affect subset is complete and invariant by model."""
    by_model = defaultdict(lambda: defaultdict(set))
    family_by_pair = defaultdict(set)
    for row in rows:
        if row.get("arm") != "intact" or row.get("gold_class") != "term":
            continue
        if row.get(field) != target:
            continue
        by_model[row["model"]][row["pair_id"]].add(row.get("framing"))
        if row.get("affect_family"):
            family_by_pair[row["pair_id"]].add(row["affect_family"])
    if not by_model:
        raise ValueError(f"no records found for affective subset {field}={target}")
    incomplete = [
        (model, pair_id)
        for model, pairs in by_model.items()
        for pair_id, framings in pairs.items()
        if framings != {"neutral", "emotional"}
    ]
    if incomplete:
        raise ValueError(f"incomplete neutral/emotional affective pairs: {incomplete[:3]}")
    counts = {model: len(pairs) for model, pairs in sorted(by_model.items())}
    if expected_pairs is not None:
        wrong = {model: n for model, n in counts.items() if n != expected_pairs}
        if wrong:
            raise ValueError(
                f"affective subset expected {expected_pairs} pairs per model; observed {wrong}"
            )
    reference = next(iter(by_model.values())).keys()
    reference = set(reference)
    nonidentical = [model for model, pairs in by_model.items() if set(pairs) != reference]
    if nonidentical:
        raise ValueError("models do not share one affective pair set: " + ", ".join(nonidentical))
    ambiguous = {pair_id: fam for pair_id, fam in family_by_pair.items() if len(fam) != 1}
    if ambiguous:
        raise ValueError(f"affect families differ within pair: {list(ambiguous)[:3]}")
    return {
        "passed": True,
        "field": field,
        "target": target,
        "pairs_per_model": counts,
        "affect_family_counts": dict(sorted(Counter(
            next(iter(family_by_pair[pair_id])) for pair_id in reference
        ).items())),
    }


def _ci(values, *, n_boot, seed):
    point, lo, hi = hierarchical_cluster_ci(
        values, n_boot=n_boot, seed=seed)
    n_pairs = sum(len(pairs) for pairs in values.values())
    n_records = sum(len(v) for pairs in values.values() for v in pairs.values())
    return {
        "estimate": round(point, 5),
        "ci95": [round(lo, 5), round(hi, 5)],
        "n_models": len(values),
        "n_model_pairs": n_pairs,
        "n_records": n_records,
    }


def _paired_rows(rows, role, control, *, arm="intact", population="term"):
    index = defaultdict(dict)
    for row in rows:
        if row["arm"] != arm or row.get("gold_class") != population:
            continue
        if row.get("role") in (role, control):
            index[(row["model"], row["record_id"])][row["role"]] = row
    for (model, _record_id), cells in index.items():
        if role in cells and control in cells:
            yield model, cells[role], cells[control]


def _role_contrast(rows, role, control, *, n_boot, seed, margin):
    disagreement = defaultdict(lambda: defaultdict(list))
    accuracy_delta = defaultdict(lambda: defaultdict(list))
    margin_delta = defaultdict(lambda: defaultdict(list))
    by_model = defaultdict(list)
    for model, treated, reference in _paired_rows(rows, role, control):
        pair_id = treated["pair_id"]
        changed = int(_label(treated) != _label(reference))
        _nested_add(disagreement, model, pair_id, changed)
        _nested_add(
            accuracy_delta,
            model,
            pair_id,
            int(_label(treated) == treated["gold_pro_id"])
            - int(_label(reference) == reference["gold_pro_id"]),
        )
        by_model[model].append(changed)
        if treated.get("label_margin") is not None and reference.get("label_margin") is not None:
            _nested_add(
                margin_delta, model, pair_id,
                treated["label_margin"] - reference["label_margin"])

    dis = _ci(disagreement, n_boot=n_boot, seed=seed)
    acc = _ci(accuracy_delta, n_boot=n_boot, seed=seed + 1)
    acc["equivalence_margin"] = [-margin, margin]
    acc["equivalent"] = bool(
        math.isfinite(acc["ci95"][0])
        and acc["ci95"][0] >= -margin
        and acc["ci95"][1] <= margin
    )
    per_model = {
        model: {"disagreement_rate": round(float(np.mean(vals)), 5), "n": len(vals)}
        for model, vals in sorted(by_model.items())
    }
    out = {
        "role": role,
        "control": control,
        "population": "term",
        "arm": "intact",
        "label_disagreement": dis,
        "paired_accuracy_difference": acc,
        "per_model": per_model,
    }
    if margin_delta:
        out["constrained_label_margin_difference"] = _ci(
            margin_delta, n_boot=n_boot, seed=seed + 2)
    else:
        out["constrained_label_margin_difference"] = {
            "available": False,
            "reason": "label_margin absent; rerun with --scorer both or constrained",
        }
    return out


def _role_by_framing(
    rows,
    role,
    control,
    *,
    n_boot,
    seed,
    subset_field=None,
    subset_value=None,
):
    """Paired 2x2 role-by-framing analysis.

    The key estimand is not a generic accuracy interaction. It asks whether the
    probability that the role changes the committed code is different for the
    emotional versus neutral member of the same clinical pair. Within-role
    emotional-vs-neutral code sensitivity and continuous margin/accuracy
    interactions are retained as secondary descriptions.
    """
    cells = defaultdict(dict)
    for row in rows:
        if row["arm"] != "intact" or row.get("gold_class") != "term":
            continue
        if subset_field and row.get(subset_field) != subset_value:
            continue
        if row.get("role") in (role, control):
            cells[(row["model"], row["pair_id"], row["role"])][row["framing"]] = row

    framed_by_role = {}
    for (model, pair_id, which), framed in cells.items():
        if "neutral" in framed and "emotional" in framed:
            framed_by_role[(model, pair_id, which)] = framed

    role_modification = defaultdict(lambda: defaultdict(list))
    accuracy_interaction = defaultdict(lambda: defaultdict(list))
    margin_interaction = defaultdict(lambda: defaultdict(list))
    within_role = {
        which: {
            "label_disagreement": defaultdict(lambda: defaultdict(list)),
            "accuracy_difference": defaultdict(lambda: defaultdict(list)),
            "label_margin_difference": defaultdict(lambda: defaultdict(list)),
        }
        for which in (role, control)
    }
    complete_pairs = 0
    for model, pair_id, which in list(framed_by_role):
        if which != role:
            continue
        a = (model, pair_id, role)
        b = (model, pair_id, control)
        if b not in framed_by_role:
            continue
        complete_pairs += 1
        role_rows, control_rows = framed_by_role[a], framed_by_role[b]
        role_dis_emotional = int(
            _label(role_rows["emotional"]) != _label(control_rows["emotional"])
        )
        role_dis_neutral = int(
            _label(role_rows["neutral"]) != _label(control_rows["neutral"])
        )
        _nested_add(
            role_modification,
            model,
            pair_id,
            role_dis_emotional - role_dis_neutral,
        )

        accuracy_delta = {}
        margin_delta = {}
        for which, framed in ((role, role_rows), (control, control_rows)):
            emotional, neutral = framed["emotional"], framed["neutral"]
            _nested_add(
                within_role[which]["label_disagreement"],
                model,
                pair_id,
                int(_label(emotional) != _label(neutral)),
            )
            accuracy_delta[which] = (
                int(_label(emotional) == emotional["gold_pro_id"])
                - int(_label(neutral) == neutral["gold_pro_id"])
            )
            _nested_add(
                within_role[which]["accuracy_difference"],
                model,
                pair_id,
                accuracy_delta[which],
            )
            if emotional.get("label_margin") is not None and neutral.get("label_margin") is not None:
                margin_delta[which] = float(emotional["label_margin"]) - float(
                    neutral["label_margin"]
                )
                _nested_add(
                    within_role[which]["label_margin_difference"],
                    model,
                    pair_id,
                    margin_delta[which],
                )
        _nested_add(
            accuracy_interaction,
            model,
            pair_id,
            accuracy_delta[role] - accuracy_delta[control],
        )
        if role in margin_delta and control in margin_delta:
            _nested_add(
                margin_interaction,
                model,
                pair_id,
                margin_delta[role] - margin_delta[control],
            )

    if not role_modification:
        return {"available": False}
    result = {
        "available": True,
        "role": role,
        "control": control,
        "subset": ({"field": subset_field, "value": subset_value}
                   if subset_field else None),
        "n_complete_model_pairs": complete_pairs,
        "role_disagreement_modification_emotional_minus_neutral": _ci(
            role_modification, n_boot=n_boot, seed=seed
        ),
        "accuracy_difference_in_differences": _ci(
            accuracy_interaction, n_boot=n_boot, seed=seed + 1
        ),
        "framing_sensitivity_by_role": {},
    }
    if margin_interaction:
        result["label_margin_difference_in_differences"] = _ci(
            margin_interaction, n_boot=n_boot, seed=seed + 2
        )
    for i, which in enumerate((role, control)):
        measures = within_role[which]
        result["framing_sensitivity_by_role"][which] = {
            "label_disagreement": _ci(
                measures["label_disagreement"], n_boot=n_boot, seed=seed + 10 + i
            ),
            "accuracy_difference_emotional_minus_neutral": _ci(
                measures["accuracy_difference"], n_boot=n_boot, seed=seed + 20 + i
            ),
        }
        if measures["label_margin_difference"]:
            result["framing_sensitivity_by_role"][which][
                "label_margin_difference_emotional_minus_neutral"
            ] = _ci(
                measures["label_margin_difference"],
                n_boot=n_boot,
                seed=seed + 30 + i,
            )
    return result


def _coding_consequences(rows, role, control, *, n_boot, seed):
    """Describe whether a role-induced code change fixes or breaks the gold code."""
    categories = {
        name: defaultdict(lambda: defaultdict(list))
        for name in (
            "unchanged_correct",
            "unchanged_incorrect",
            "corrected_by_role",
            "broken_by_role",
            "changed_wrong_to_wrong",
        )
    }
    for model, treated, reference in _paired_rows(rows, role, control):
        pair_id = treated["pair_id"]
        treated_correct = _label(treated) == treated["gold_pro_id"]
        reference_correct = _label(reference) == reference["gold_pro_id"]
        changed = _label(treated) != _label(reference)
        if not changed and treated_correct:
            category = "unchanged_correct"
        elif not changed:
            category = "unchanged_incorrect"
        elif treated_correct and not reference_correct:
            category = "corrected_by_role"
        elif reference_correct and not treated_correct:
            category = "broken_by_role"
        else:
            category = "changed_wrong_to_wrong"
        for name, values in categories.items():
            _nested_add(values, model, pair_id, int(name == category))
    if not any(categories.values()):
        return {"available": False}
    return {
        "available": True,
        "role": role,
        "control": control,
        "mutually_exclusive_rates": {
            name: _ci(values, n_boot=n_boot, seed=seed + i)
            for i, (name, values) in enumerate(categories.items())
        },
        "interpretation": (
            "Code changes are partitioned into corrected, broken and wrong-to-wrong "
            "transitions; no unvalidated clinical-severity ordering is imposed on PRO terms."
        ),
    }


def _medicalization_contrast(
    rows,
    model_pairs,
    role,
    control,
    *,
    n_boot,
    seed,
    subset_field=None,
    subset_value=None,
):
    """Matched base-vs-medicalized contrast over architecture families."""
    index = defaultdict(dict)
    wanted = {role, control}
    for row in rows:
        if row.get("arm") != "intact" or row.get("gold_class") != "term":
            continue
        if subset_field and row.get(subset_field) != subset_value:
            continue
        if row.get("role") in wanted:
            index[(row["model"], row["record_id"])][row["role"]] = row

    disagreement_delta = defaultdict(lambda: defaultdict(list))
    framing_modification_delta = defaultdict(lambda: defaultdict(list))
    per_family = {}
    for pair in model_pairs:
        family = pair["family"]
        base, medicalized = _slug(pair["base"]), _slug(pair["medicalized"])
        per_record = {}
        by_pair = defaultdict(dict)
        for model_slug, label in ((base, "base"), (medicalized, "medicalized")):
            record_ids = {record_id for model, record_id in index if model == model_slug}
            for record_id in record_ids:
                cells = index[(model_slug, record_id)]
                if set(cells) != wanted:
                    continue
                treated, reference = cells[role], cells[control]
                value = int(_label(treated) != _label(reference))
                per_record[(label, record_id)] = value
                by_pair[(label, treated["pair_id"])][treated["framing"]] = value

        common_records = sorted(
            record_id
            for label, record_id in per_record
            if label == "base" and ("medicalized", record_id) in per_record
        )
        for record_id in common_records:
            pair_id = index[(base, record_id)][role]["pair_id"]
            _nested_add(
                disagreement_delta,
                family,
                pair_id,
                per_record[("medicalized", record_id)] - per_record[("base", record_id)],
            )

        family_modifications = []
        common_pair_ids = sorted(
            pair_id
            for label, pair_id in by_pair
            if label == "base" and ("medicalized", pair_id) in by_pair
        )
        for pair_id in common_pair_ids:
            base_frames = by_pair[("base", pair_id)]
            med_frames = by_pair[("medicalized", pair_id)]
            if not {"neutral", "emotional"}.issubset(base_frames) or not {
                "neutral", "emotional"
            }.issubset(med_frames):
                continue
            base_mod = base_frames["emotional"] - base_frames["neutral"]
            med_mod = med_frames["emotional"] - med_frames["neutral"]
            delta = med_mod - base_mod
            family_modifications.append(delta)
            _nested_add(framing_modification_delta, family, pair_id, delta)

        base_values = [per_record[("base", record_id)] for record_id in common_records]
        med_values = [per_record[("medicalized", record_id)] for record_id in common_records]
        per_family[family] = {
            "base_model": pair["base"],
            "medicalized_model": pair["medicalized"],
            "n_common_records": len(common_records),
            "base_role_disagreement": (
                round(float(np.mean(base_values)), 5) if base_values else None
            ),
            "medicalized_role_disagreement": (
                round(float(np.mean(med_values)), 5) if med_values else None
            ),
            "mean_framing_modification_difference": (
                round(float(np.mean(family_modifications)), 5)
                if family_modifications else None
            ),
        }

    if not disagreement_delta:
        return {"available": False, "reason": "no complete base/medicalized pairs"}
    result = {
        "available": True,
        "n_families": len(per_family),
        "subset": ({"field": subset_field, "value": subset_value}
                   if subset_field else None),
        "role_disagreement_difference_medicalized_minus_base": _ci(
            disagreement_delta, n_boot=n_boot, seed=seed
        ),
        "per_family": per_family,
        "interpretation": (
            "Secondary matched-family estimate; three families do not support a broad "
            "claim about all medical fine-tuning."
        ),
    }
    if framing_modification_delta:
        result[
            "role_by_framing_modification_difference_medicalized_minus_base"
        ] = _ci(framing_modification_delta, n_boot=n_boot, seed=seed + 1)
    return result


def _mechanistic_gate(
    rows,
    role,
    control,
    target_arm,
    random_arm,
    *,
    n_boot,
    seed,
    subset_field=None,
    subset_value=None,
):
    index = defaultdict(dict)
    wanted = {"intact", target_arm, random_arm}
    for row in rows:
        if row.get("gold_class") != "term" or row["arm"] not in wanted:
            continue
        if subset_field and row.get(subset_field) != subset_value:
            continue
        if row.get("role") in (role, control):
            index[(row["model"], row["record_id"], row["arm"])][row["role"]] = row

    by_record = defaultdict(dict)
    for (model, record_id, arm), role_rows in index.items():
        if role in role_rows and control in role_rows:
            by_record[(model, record_id)][arm] = int(
                _label(role_rows[role]) != _label(role_rows[control]))
            by_record[(model, record_id)]["pair_id"] = role_rows[role]["pair_id"]

    targeted = defaultdict(lambda: defaultdict(list))
    random = defaultdict(lambda: defaultdict(list))
    advantage = defaultdict(lambda: defaultdict(list))
    rates = {arm: defaultdict(lambda: defaultdict(list)) for arm in wanted}
    complete = 0
    for (model, record_id), arms in by_record.items():
        if not wanted.issubset(arms):
            continue
        complete += 1
        pair_id = arms["pair_id"]
        for arm in wanted:
            _nested_add(rates[arm], model, pair_id, arms[arm])
        _nested_add(targeted, model, pair_id, arms["intact"] - arms[target_arm])
        _nested_add(random, model, pair_id, arms["intact"] - arms[random_arm])
        # Positive: targeted ablation leaves less cross-role disagreement than
        # norm/layer-matched random ablation.
        _nested_add(advantage, model, pair_id, arms[random_arm] - arms[target_arm])

    if not complete:
        return {
            "available": False,
            "reason": "no records complete across both roles and all three arms",
        }
    target_ci = _ci(targeted, n_boot=n_boot, seed=seed)
    random_ci = _ci(random, n_boot=n_boot, seed=seed + 1)
    advantage_ci = _ci(advantage, n_boot=n_boot, seed=seed + 2)
    return {
        "available": True,
        "role": role,
        "control": control,
        "subset": ({"field": subset_field, "value": subset_value}
                   if subset_field else None),
        "n_complete_records": complete,
        "role_disagreement_by_arm": {
            arm: _ci(values, n_boot=n_boot, seed=seed + 10 + i)
            for i, (arm, values) in enumerate(sorted(rates.items()))
        },
        "targeted_attenuation": target_ci,
        "random_attenuation": random_ci,
        "attenuation_advantage_targeted_vs_random": advantage_ci,
        "gate_passes": bool(
            target_ci["ci95"][0] > 0 and advantage_ci["ci95"][0] > 0
        ),
    }


def _generative_safety(rows, roles):
    out = {}
    for role in roles:
        subset = [r for r in rows if r["arm"] == "intact"
                  and r.get("gold_class") == "abstain" and r.get("role") == role]
        kinds = [_generative_kind(r) for r in subset]
        available = [k for k in kinds if k is not None]
        if not available:
            out[role] = {
                "available": False,
                "reason": "free-generation outcome absent; constrained scoring cannot measure abstention",
            }
            continue
        counts = Counter(available)
        n = len(available)
        out[role] = {
            "available": True,
            "n": n,
            "mapped_false_positive_rate": round(counts["mapped"] / n, 5),
            "abstention_rate": round(counts["abstained"] / n, 5),
            "non_answer_rate": round(counts["non_answer"] / n, 5),
            "unmapped_rate": round(counts["unmapped"] / n, 5),
        }
    return out


def _affective_profile(
    rows,
    role,
    control,
    concepts,
    *,
    n_boot,
    seed,
    score_field="z",
    groups=None,
    eligible_axes_by_model=None,
    subset_field=None,
    subset_value=None,
):
    deltas = {c: defaultdict(lambda: defaultdict(list)) for c in concepts}
    distance = defaultdict(lambda: defaultdict(list))
    changed_distance = []
    stable_distance = []
    association_by_model = defaultdict(list)
    record_deltas = {}
    group_delta = {
        name: defaultdict(lambda: defaultdict(list)) for name in (groups or {})
    }
    used = []
    for model, treated, reference in _paired_rows(rows, role, control):
        if subset_field and treated.get(subset_field) != subset_value:
            continue
        z_a, z_b = treated.get(score_field) or {}, reference.get(score_field) or {}
        eligible = None
        if eligible_axes_by_model is not None:
            eligible = set(eligible_axes_by_model.get(model, []))
        available = [c for c in concepts if c in z_a and c in z_b]
        if eligible is not None:
            available = [c for c in available if c in eligible]
        if not available:
            continue
        used.extend(available)
        diffs = []
        per_record = {}
        for concept in available:
            delta = float(z_a[concept]) - float(z_b[concept])
            diffs.append(delta)
            per_record[concept] = delta
            _nested_add(deltas[concept], model, treated["pair_id"], delta)
        rms = float(np.sqrt(np.mean(np.square(diffs))))
        _nested_add(distance, model, treated["pair_id"], rms)
        changed = int(_label(treated) != _label(reference))
        (changed_distance if changed else stable_distance).append(rms)
        association_by_model[model].append((rms, changed))
        record_deltas[(model, treated["pair_id"], treated["framing"])] = per_record
        for name, members in (groups or {}).items():
            values = [per_record[c] for c in members if c in per_record]
            if values:
                _nested_add(group_delta[name], model, treated["pair_id"], np.mean(values))

    if not distance:
        return {"available": False, "readout": score_field}

    framing_group_interaction = {
        name: defaultdict(lambda: defaultdict(list)) for name in (groups or {})
    }
    for (model, pair_id, framing), per_record in record_deltas.items():
        if framing != "emotional":
            continue
        neutral = record_deltas.get((model, pair_id, "neutral"))
        if neutral is None:
            continue
        for name, members in (groups or {}).items():
            common = [c for c in members if c in per_record and c in neutral]
            if common:
                interaction = np.mean([per_record[c] - neutral[c] for c in common])
                _nested_add(framing_group_interaction[name], model, pair_id, interaction)

    correlations = {}
    for model, values in association_by_model.items():
        rms = np.asarray([v[0] for v in values], dtype=float)
        changed = np.asarray([v[1] for v in values], dtype=float)
        if len(values) >= 3 and np.std(rms) > 0 and np.std(changed) > 0:
            correlations[model] = round(float(np.corrcoef(rms, changed)[0, 1]), 5)

    result = {
        "available": True,
        "role": role,
        "control": control,
        "readout": score_field,
        "subset": ({"field": subset_field, "value": subset_value}
                   if subset_field else None),
        "concepts_requested": concepts,
        "concepts_available": sorted(set(used)),
        "eligible_axes_by_model": eligible_axes_by_model,
        "profile_rms_shift": _ci(distance, n_boot=n_boot, seed=seed),
        "per_axis_delta": {
            concept: _ci(values, n_boot=n_boot, seed=seed + i + 1)
            for i, (concept, values) in enumerate(deltas.items()) if values
        },
        "rms_shift_changed_labels": (
            round(float(np.mean(changed_distance)), 5) if changed_distance else None),
        "rms_shift_stable_labels": (
            round(float(np.mean(stable_distance)), 5) if stable_distance else None),
        "point_biserial_rms_shift_vs_label_churn_by_model": correlations,
        "mean_point_biserial_correlation": (
            round(float(np.mean(list(correlations.values()))), 5)
            if correlations else None
        ),
        "interpretation_gate": (
            "Only model-axis cells passing the preregistered out-of-fold AUROC and "
            "lexical-cosine thresholds contribute; association with label churn is "
            "descriptive and is not a mediation test."
        ),
    }
    if groups:
        result["per_group_role_delta"] = {
            name: _ci(values, n_boot=n_boot, seed=seed + 100 + i)
            for i, (name, values) in enumerate(group_delta.items()) if values
        }
        result["role_by_framing_group_interaction"] = {
            name: _ci(values, n_boot=n_boot, seed=seed + 200 + i)
            for i, (name, values) in enumerate(framing_group_interaction.items())
            if values
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-glob", default=str(_ROOT / "outputs/role_emotion/*__rows.jsonl"))
    parser.add_argument("--study-config", type=Path,
                        default=_ROOT / "configs/study_esmo_2026.yaml")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "outputs/reports/esmo_primary_analysis.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.rows_glob))
    if not paths:
        print(f"no rows files matched {args.rows_glob}")
        return 1
    config = yaml.safe_load(args.study_config.read_text(encoding="utf-8"))
    selected_paths, excluded_models, artifact_validation = _validated_tier_paths(
        paths, config, args.study_config)
    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        current_commit = None
    if current_commit and artifact_validation["git_commit"] != current_commit:
        raise ValueError(
            "Tier 1 rows were produced by git commit "
            f"{artifact_validation['git_commit']}, but analysis is running at {current_commit}"
        )
    primary_cfg = config["primary"]
    role, control = primary_cfg["role"], primary_cfg["control"]
    n_boot, seed = int(primary_cfg["bootstrap_draws"]), int(primary_cfg["seed"])
    margin = float(config["accuracy_equivalence"]["margin"])
    rows = _load(selected_paths)
    roles_present = sorted({r["role"] for r in rows})
    missing = sorted({role, control} - set(roles_present))
    if missing:
        raise ValueError(
            f"definitive primary contrast cannot run; missing roles {missing}. "
            "These are legacy rows and must be regenerated with the current protocol."
        )
    quality = _validate_primary_rows(
        rows,
        role,
        control,
        expected_records=primary_cfg.get("expected_term_records_per_model"),
    )
    framing_cfg = config["affective_framing"]
    affective_quality = _validate_affective_subset(
        rows,
        field=framing_cfg["subset_field"],
        target=framing_cfg["target_subset"],
        expected_pairs=framing_cfg.get("expected_target_pairs"),
    )

    primary = _role_contrast(
        rows, role, control, n_boot=n_boot, seed=seed, margin=margin)
    secondary = {}
    for i, (a, b) in enumerate(config.get("secondary_contrasts", [])):
        if a in roles_present and b in roles_present:
            secondary[f"{a}_vs_{b}"] = _role_contrast(
                rows, a, b, n_boot=n_boot, seed=seed + 100 * (i + 1), margin=margin)

    gate_cfg = config["mechanistic_gate"]
    mechanism = _mechanistic_gate(
        rows, role, control,
        gate_cfg["target_arm"], gate_cfg["control_arm"],
        n_boot=n_boot, seed=seed + 1000)
    subset_field = framing_cfg["subset_field"]
    target_subset = framing_cfg["target_subset"]
    control_subset = framing_cfg["specificity_control_subset"]
    role_framing = _role_by_framing(
        rows, role, control, n_boot=n_boot, seed=seed + 2000,
        subset_field=subset_field, subset_value=target_subset)
    role_framing_specificity = _role_by_framing(
        rows, role, control, n_boot=n_boot, seed=seed + 2100,
        subset_field=subset_field, subset_value=control_subset)
    mechanism_by_manipulation = {
        subset: _mechanistic_gate(
            rows, role, control,
            gate_cfg["target_arm"], gate_cfg["control_arm"],
            n_boot=n_boot, seed=seed + 2200 + i * 100,
            subset_field=subset_field, subset_value=subset)
        for i, subset in enumerate((target_subset, control_subset))
    }
    profile_cfg = config["affective_profile"]
    eligible_axes = artifact_validation.get("eligible_affect_axes_by_model", {})
    profile_subset_field = profile_cfg.get("population_subset_field")
    profile_subset_value = profile_cfg.get("population_subset_value")
    affect = _affective_profile(
        rows, role, control, profile_cfg["concepts"],
        n_boot=n_boot, seed=seed + 3000, score_field="z",
        groups=profile_cfg.get("groups"), eligible_axes_by_model=eligible_axes,
        subset_field=profile_subset_field, subset_value=profile_subset_value)
    affect_read = _affective_profile(
        rows, role, control, profile_cfg["concepts"],
        n_boot=n_boot, seed=seed + 3500, score_field="z_read",
        groups=profile_cfg.get("groups"), eligible_axes_by_model=eligible_axes,
        subset_field=profile_subset_field, subset_value=profile_subset_value)
    consequences = _coding_consequences(
        rows, role, control, n_boot=n_boot, seed=seed + 4000)
    medicalization_all = _medicalization_contrast(
        rows, config.get("model_pairs", []), role, control,
        n_boot=n_boot, seed=seed + 5000)
    medicalization_affect = _medicalization_contrast(
        rows, config.get("model_pairs", []), role, control,
        n_boot=n_boot, seed=seed + 5100,
        subset_field=subset_field, subset_value=target_subset)
    safety = _generative_safety(rows, roles_present)

    dis = primary["label_disagreement"]
    acc = primary["paired_accuracy_difference"]
    affective_modification = role_framing.get(
        "role_disagreement_modification_emotional_minus_neutral", {}
    )
    affective_ci = affective_modification.get("ci95", [math.nan, math.nan])
    affective_signal = bool(
        all(math.isfinite(v) for v in affective_ci)
        and (affective_ci[0] > 0 or affective_ci[1] < 0)
    )
    readiness = {
        "behavioral_signal_detected": bool(dis["ci95"][0] > 0),
        "accuracy_equivalence_supported": acc["equivalent"],
        "affective_role_modification_detected": affective_signal,
        "mechanistic_gate_passes": mechanism.get("gate_passes"),
        "two_legged_affective_claim_ready": bool(
            dis["ci95"][0] > 0 and acc["equivalent"]
            and mechanism.get("gate_passes") is True),
        "fallback_if_mechanistic_gate_fails": (
            "Report role-conditioned coding instability as a behavioral safety finding; "
            "do not attribute it causally to affect."
        ),
    }
    report = {
        "protocol_id": config["protocol_id"],
        "models": sorted({r["model"] for r in rows}),
        "model_ids": {_slug(model_id): model_id for model_id in config["models"]["tier1"]},
        "excluded_non_tier1_models": excluded_models,
        "roles": roles_present,
        "n_rows": len(rows),
        "artifact_validation": artifact_validation,
        "data_quality": {**quality, "affective_subset": affective_quality},
        "primary": primary,
        "accuracy_invisibility_test": primary["paired_accuracy_difference"],
        "mechanistic_gate": mechanism,
        "mechanistic_sensitivity_by_manipulation": mechanism_by_manipulation,
        "affective_profile": affect,
        "affective_profile_read_point": affect_read,
        "affective_framing_key_secondary": role_framing,
        "symptom_intensity_specificity_control": role_framing_specificity,
        # Compatibility alias for older report consumers.
        "role_by_framing_secondary": role_framing,
        "coding_transition_consequences": consequences,
        "medicalization_secondary": {
            "all_terms": medicalization_all,
            "affective_reaction": medicalization_affect,
        },
        "secondary_role_contrasts": secondary,
        "generative_abstention_safety": safety,
        "abstract_readiness": readiness,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(rows)} rows | {len(report['models'])} models | protocol={config['protocol_id']}")
    print("\nPRIMARY — role-induced label disagreement")
    print(f"  {role} vs {control}: {dis['estimate']:.1%} "
          f"[{dis['ci95'][0]:.1%}, {dis['ci95'][1]:.1%}]  n={dis['n_records']}")
    print("\nACCURACY EQUIVALENCE")
    print(f"  paired difference {acc['estimate']:+.3f} "
          f"[{acc['ci95'][0]:+.3f}, {acc['ci95'][1]:+.3f}] "
          f"within ±{margin:.3f}: {acc['equivalent']}")
    print("\nAFFECTIVE FRAMING x ROLE")
    if role_framing.get("available"):
        mod = role_framing[
            "role_disagreement_modification_emotional_minus_neutral"
        ]
        print(
            f"  change in cross-role disagreement {mod['estimate']:+.3f} "
            f"[{mod['ci95'][0]:+.3f}, {mod['ci95'][1]:+.3f}] "
            f"on {target_subset} pairs"
        )
    else:
        print("  unavailable")
    print("\nMECHANISTIC GATE")
    if mechanism.get("available"):
        adv = mechanism["attenuation_advantage_targeted_vs_random"]
        print(f"  targeted attenuation advantage {adv['estimate']:+.3f} "
              f"[{adv['ci95'][0]:+.3f}, {adv['ci95'][1]:+.3f}] "
              f"PASS={mechanism['gate_passes']}")
    else:
        print(f"  unavailable: {mechanism.get('reason')}")
    print(f"\nWrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
