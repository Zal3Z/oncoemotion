"""Regression tests for the preregistered ESMO analysis contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "analyze_results.py"
_ABSTRACT_SCRIPT = _ROOT / "scripts" / "build_esmo_abstract.py"
_POSTER_SCRIPT = _ROOT / "scripts" / "build_esmo_poster_figures.py"


def _analysis():
    spec = importlib.util.spec_from_file_location("_esmo_analysis", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _abstract_builder():
    spec = importlib.util.spec_from_file_location("_esmo_abstract", _ABSTRACT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _poster_builder():
    spec = importlib.util.spec_from_file_location("_esmo_poster", _POSTER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(model, record, pair, role, arm, label, gold="PRO_001", **extra):
    return {
        "model": model,
        "record_id": record,
        "pair_id": pair,
        "role": role,
        "arm": arm,
        "ablated": arm != "intact",
        "framing": "neutral" if record.endswith("n") else "emotional",
        "manipulation_type": "affective_reaction",
        "affect_family": "threat",
        "gold_class": "term",
        "gold_pro_id": gold,
        "model_top1_id": label,
        "scorer": "constrained",
        "label_margin": 0.2,
        "z": {"afraid_alarmed": 0.0},
        **extra,
    }


def test_primary_is_label_disagreement_while_accuracy_can_be_equivalent():
    mod = _analysis()
    rows = []
    for model in ("m1", "m2"):
        # The code changes on both framings. Accuracy gains one item and loses one,
        # so aggregate accuracy is exactly unchanged: the intended safety finding.
        rows += [
            _row(model, "r1n", "p1", "oncologo", "intact", "PRO_001"),
            _row(model, "r1n", "p1", "none_filler", "intact", "PRO_002"),
            _row(model, "r1e", "p1", "oncologo", "intact", "PRO_002"),
            _row(model, "r1e", "p1", "none_filler", "intact", "PRO_001"),
        ]
    result = mod._role_contrast(
        rows, "oncologo", "none_filler", n_boot=200, seed=7, margin=0.05)
    assert result["label_disagreement"]["estimate"] == 1.0
    assert result["paired_accuracy_difference"]["estimate"] == 0.0
    assert result["paired_accuracy_difference"]["equivalent"] is True


def test_mechanistic_gate_tests_attenuation_of_the_role_effect():
    mod = _analysis()
    rows = []
    for model in ("m1", "m2"):
        for record in ("r1n", "r1e"):
            pair = "p1"
            # Intact and random arms disagree across roles. Targeted emotion
            # ablation makes the two roles agree.
            for arm, treated in (("intact", "PRO_002"),
                                 ("emotion", "PRO_001"),
                                 ("random", "PRO_002")):
                rows.append(_row(model, record, pair, "oncologo", arm, treated))
                rows.append(_row(model, record, pair, "none_filler", arm, "PRO_001"))
    gate = mod._mechanistic_gate(
        rows, "oncologo", "none_filler", "emotion", "random",
        n_boot=200, seed=5)
    assert gate["targeted_attenuation"]["estimate"] == 1.0
    assert gate["random_attenuation"]["estimate"] == 0.0
    assert gate["attenuation_advantage_targeted_vs_random"]["estimate"] == 1.0
    assert gate["gate_passes"] is True


def test_constrained_codes_are_not_misreported_as_abstention_failures():
    mod = _analysis()
    base = _row("m1", "r1n", "p1", "oncologo", "intact", "PRO_001")
    base["gold_class"] = "abstain"
    unavailable = mod._generative_safety([base], ["oncologo"])["oncologo"]
    assert unavailable["available"] is False

    base["generative_kind"] = "abstained"
    available = mod._generative_safety([base], ["oncologo"])["oncologo"]
    assert available["available"] is True
    assert available["abstention_rate"] == 1.0
    assert available["mapped_false_positive_rate"] == 0.0


def test_primary_quality_gate_rejects_duplicate_rows():
    mod = _analysis()
    row = _row("m1", "r1n", "p1", "oncologo", "intact", "PRO_001")
    control = _row("m1", "r1n", "p1", "none_filler", "intact", "PRO_001")
    with pytest.raises(ValueError, match="duplicate"):
        mod._validate_primary_rows([row, row.copy(), control], "oncologo", "none_filler")


def test_primary_quality_gate_accepts_complete_common_record_set():
    mod = _analysis()
    rows = []
    for model in ("m1", "m2"):
        rows.extend([
            _row(model, "r1n", "p1", "oncologo", "intact", "PRO_001"),
            _row(model, "r1n", "p1", "none_filler", "intact", "PRO_001"),
        ])
    result = mod._validate_primary_rows(
        rows, "oncologo", "none_filler", expected_records=1)
    assert result["passed"] is True
    assert result["term_records_per_model"] == {"m1": 1, "m2": 1}


def test_tier_artifact_gate_validates_metadata_and_rows_hash(tmp_path):
    mod = _analysis()
    model_ids = ["org/model-a", "org/model-b"]
    config = {
        "protocol_id": "p1",
        "models": {"tier1": model_ids},
        "mechanistic_gate": {"target_axes": ["afraid_alarmed"]},
        "design": {
            "roles": ["oncologo", "none_filler"],
            "arms": ["intact", "emotion", "random"],
            "expected_total_records_per_model": 1,
        },
    }
    study = tmp_path / "study.yaml"
    study.write_text("protocol_id: p1\n", encoding="utf-8")
    paths = []
    for model_id in model_ids:
        slug = mod._slug(model_id)
        rows = tmp_path / f"{slug}__rows.jsonl"
        rows.write_text("{}\n", encoding="utf-8")
        meta = {
            "model_id": model_id,
            "protocol_id": "p1",
            "study_config_sha256": mod._sha256(study),
            "roles": config["design"]["roles"],
            "arms_requested": config["design"]["arms"],
            "scorer": "both",
            "n_items": 1,
            "rows_sha256": mod._sha256(rows),
            "dataset_sha256": "same-dataset",
            "git_commit": "abc123",
            "ablate_concepts_requested": ["afraid_alarmed"],
            "eligible_affect_axes": ["afraid_alarmed"],
        }
        rows.with_name(f"{slug}__meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
        paths.append(rows)

    selected, excluded, result = mod._validated_tier_paths(paths, config, study)
    assert len(selected) == 2 and excluded == []
    assert result["passed"] is True

    paths[0].write_text("{}\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rows_sha256"):
        mod._validated_tier_paths(paths, config, study)


def test_affective_role_by_framing_uses_label_instability_not_only_accuracy():
    mod = _analysis()
    rows = []
    for model in ("m1", "m2"):
        rows.extend([
            _row(model, "r1n", "p1", "oncologo", "intact", "PRO_001"),
            _row(model, "r1n", "p1", "none_filler", "intact", "PRO_001"),
            _row(model, "r1e", "p1", "oncologo", "intact", "PRO_002"),
            _row(model, "r1e", "p1", "none_filler", "intact", "PRO_001"),
        ])
    result = mod._role_by_framing(
        rows,
        "oncologo",
        "none_filler",
        n_boot=200,
        seed=11,
        subset_field="manipulation_type",
        subset_value="affective_reaction",
    )
    assert result["role_disagreement_modification_emotional_minus_neutral"][
        "estimate"
    ] == 1.0
    assert result["framing_sensitivity_by_role"]["oncologo"][
        "label_disagreement"
    ]["estimate"] == 1.0
    assert result["framing_sensitivity_by_role"]["none_filler"][
        "label_disagreement"
    ]["estimate"] == 0.0


def test_affective_subset_gate_requires_complete_paired_framings():
    mod = _analysis()
    rows = [
        _row("m1", "r1n", "p1", "oncologo", "intact", "PRO_001"),
        _row("m1", "r1e", "p1", "oncologo", "intact", "PRO_001"),
    ]
    result = mod._validate_affective_subset(
        rows,
        field="manipulation_type",
        target="affective_reaction",
        expected_pairs=1,
    )
    assert result["passed"] is True
    assert result["affect_family_counts"] == {"threat": 1}

    with pytest.raises(ValueError, match="incomplete"):
        mod._validate_affective_subset(
            rows[:1],
            field="manipulation_type",
            target="affective_reaction",
            expected_pairs=1,
        )


def test_medicalization_contrast_is_matched_within_model_family():
    mod = _analysis()
    rows = []
    base = "base-a"
    medicalized = "medical-a"
    for model, emotional_role_label in ((base, "PRO_001"), (medicalized, "PRO_002")):
        rows.extend([
            _row(model, "r1n", "p1", "oncologo", "intact", "PRO_001"),
            _row(model, "r1n", "p1", "none_filler", "intact", "PRO_001"),
            _row(model, "r1e", "p1", "oncologo", "intact", emotional_role_label),
            _row(model, "r1e", "p1", "none_filler", "intact", "PRO_001"),
        ])
    result = mod._medicalization_contrast(
        rows,
        [{
            "family": "family-a",
            "base": "org/base-a",
            "medicalized": "org/medical-a",
        }],
        "oncologo",
        "none_filler",
        n_boot=200,
        seed=17,
        subset_field="manipulation_type",
        subset_value="affective_reaction",
    )
    assert result["role_disagreement_difference_medicalized_minus_base"][
        "estimate"
    ] == 0.5
    assert result[
        "role_by_framing_modification_difference_medicalized_minus_base"
    ]["estimate"] == 1.0


def test_abstract_builder_stays_within_esmo_character_limit():
    mod = _abstract_builder()
    report = {
        "models": [f"m{i}" for i in range(9)],
        "primary": {
            "label_disagreement": {"estimate": 0.21, "ci95": [0.17, 0.25],
                                     "n_records": 2016},
            "paired_accuracy_difference": {"estimate": 0.01, "ci95": [-0.02, 0.03],
                                             "equivalent": True},
        },
        "mechanistic_gate": {
            "available": True,
            "gate_passes": True,
            "attenuation_advantage_targeted_vs_random": {
                "estimate": 0.08, "ci95": [0.02, 0.13]},
        },
        "affective_profile": {
            "available": True,
            "profile_rms_shift": {"estimate": 0.74},
        },
        "affective_framing_key_secondary": {
            "role_disagreement_modification_emotional_minus_neutral": {
                "estimate": 0.08,
                "ci95": [0.02, 0.14],
            },
        },
        "symptom_intensity_specificity_control": {
            "role_disagreement_modification_emotional_minus_neutral": {
                "estimate": 0.01,
                "ci95": [-0.03, 0.05],
            },
        },
        "abstract_readiness": {"affective_role_modification_detected": True},
    }
    config = {
        "title": (
            "Affective representations and role-conditioned instability in "
            "LLM-based PRO-CTCAE coding"
        ),
        "reporting": {"abstract_character_limit_excluding_spaces": 2000},
    }
    title, body, count = mod.build(report, config)
    assert title and "Background:" in body and "Conclusions:" in body
    assert count <= 2000


def test_abstract_does_not_describe_an_unavailable_causal_test_as_failed():
    mod = _abstract_builder()
    report = {
        "models": ["m"],
        "primary": {
            "label_disagreement": {"estimate": 0.1, "ci95": [0.05, 0.15], "n_records": 10},
            "paired_accuracy_difference": {
                "estimate": 0.0, "ci95": [-0.01, 0.01], "equivalent": True,
            },
        },
        "mechanistic_gate": {"available": False},
        "affective_profile": {
            "available": True,
            "profile_rms_shift": {"estimate": 0.2},
        },
    }
    _, body, _ = mod.build(report, {})
    assert "causal comparison was unavailable" in body
    assert "did not clear the matched random control" not in body


def test_poster_data_omits_unavailable_mechanistic_panel():
    mod = _poster_builder()
    report = {
        "primary": {
            "label_disagreement": {"estimate": 0.2, "ci95": [0.1, 0.3]},
            "paired_accuracy_difference": {
                "estimate": 0.0,
                "ci95": [-0.02, 0.02],
                "equivalence_margin": [-0.05, 0.05],
                "equivalent": True,
            },
            "per_model": {"m1": {"disagreement_rate": 0.2, "n": 224}},
        },
        "mechanistic_gate": {"available": False},
    }
    data = mod.poster_data(report)
    assert data["models"] == [{"name": "m1", "estimate": 0.2, "n": 224}]
    assert data["mechanism"] is None
