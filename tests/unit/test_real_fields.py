from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from oncoemotion.emotion_vectors.dataset import build_dataset

_ROOT = Path(__file__).resolve().parents[2]


def _script(name: str):
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_ingestion_keeps_distinct_assessments_for_identical_text():
    ingest = _script("ingest_real_fields.py")
    rows = [
        {
            "_source_row": 2,
            "campo_aperto": "Dolore",
            "valore_associato": 2,
            "fonte_associazione": "PRO-CTCAE",
            "item_associato": "Pain",
        },
        {
            "_source_row": 3,
            "campo_aperto": "Dolore ",
            "valore_associato": 4,
            "fonte_associazione": "PRO-CTCAE",
            "item_associato": "Pain",
        },
    ]
    records = ingest.convert_rows(
        rows,
        {"pain": {"canonical_id": "PRO_048"}},
        {"PRO_048": {"canonical_english": "General pain"}},
    )
    assert len(records) == 2
    assert records[0]["record_id"] != records[1]["record_id"]
    assert records[0]["pair_id"] != records[1]["pair_id"]
    assert records[0]["source_id"] == records[1]["source_id"]
    assert ingest.duplicate_audit(records)["conflicting_annotation_groups"] == 1
    with pytest.raises(ValueError, match="conflicting"):
        ingest.deduplicate_identical_annotations(records)


def test_real_ingestion_fails_closed_on_unmapped_pro_label():
    ingest = _script("ingest_real_fields.py")
    with pytest.raises(ValueError, match="unmapped PRO-CTCAE"):
        ingest.convert_rows(
            [
                {
                    "campo_aperto": "Sintomo",
                    "valore_associato": 1,
                    "fonte_assoczione": "PRO-CTCAE",
                    "fonte_associazione": "PRO-CTCAE",
                    "item_associato": "Unknown",
                }
            ],
            {},
            {},
        )


def test_augmented_emotion_dataset_has_group_separation_and_depth():
    dataset = build_dataset(seed=12345)
    counts = Counter(example.concept for example in dataset)
    for concept in (
        "afraid_alarmed",
        "anxious_nervous",
        "concerned",
        "sad",
        "anger",
        "calm",
        "hope",
        "relief",
    ):
        assert counts[concept] >= 70
    family_splits = defaultdict(set)
    for example in dataset:
        family_splits[(example.concept, example.family)].add(example.split)
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert len({example.text for example in dataset}) == len(dataset)
    assert any(example.source == "structured_augmentation" for example in dataset)


def test_cross_validation_folds_keep_paraphrase_families_together():
    validate = _script("validate_vectors.py")
    concepts = np.array(["fear"] * 6 + ["calm"] * 6)
    families = np.array(
        [
            "fear-a",
            "fear-a",
            "fear-b",
            "fear-b",
            "fear-c",
            "fear-c",
            "calm-a",
            "calm-a",
            "calm-b",
            "calm-b",
            "calm-c",
            "calm-c",
        ],
        dtype=object,
    )
    folds = validate._stratified_folds(concepts, 3, seed=7, families=families)
    for family in set(families):
        assert len(set(folds[families == family])) == 1


def test_real_primary_residual_association_recovers_direction():
    analysis = _script("analyze_real_fields.py")
    rows = []
    for index in range(80):
        exposure = ((index * 7) % 17) / 17
        rows.append(
            {
                "source_item": f"item-{index % 4}",
                "grade": index % 5,
                "n_words": 3 + index % 7,
                "x": exposure,
                "y": exposure > 0.45,
            }
        )
    slope = analysis._residual_association(
        rows,
        exposure=lambda row: row["x"],
        outcome=lambda row: float(row["y"]),
        group=lambda row: (row["source_item"], row["grade"]),
        nuisances=[lambda row: np.log1p(row["n_words"])],
    )
    assert slope > 0


def test_real_hierarchical_bootstrap_is_deterministic():
    analysis = _script("analyze_real_fields.py")
    rows = [
        {"model": model, "source_id": f"s{i}", "value": value}
        for model, value in (("m1", 0.0), ("m2", 1.0))
        for i in range(8)
    ]
    statistic = lambda selected: float(np.mean([row["value"] for row in selected]))
    a = analysis.hierarchical_cluster_stat_ci(rows, statistic, n_boot=200, seed=9)
    b = analysis.hierarchical_cluster_stat_ci(rows, statistic, n_boot=200, seed=9)
    assert a == b
    assert a["estimate"] == 0.5


def test_result_packager_rejects_raw_text(tmp_path):
    package = _script("package_real_results.py")
    path = tmp_path / "model__rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "testo clinico",
                "source_id": "abc",
                "text_redacted": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw or unverified"):
        package._assert_redacted(path)

    path.write_text(
        json.dumps(
            {
                "text": "abc",
                "source_id": "abc",
                "text_redacted": True,
                "model_generated": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    package._assert_redacted(path)


def test_reasoning_packager_rejects_generated_deliberation(tmp_path):
    package = _script("package_reasoning_results.py")
    path = tmp_path / "model__rows.jsonl"
    base = {
        "text": "abc",
        "source_id": "abc",
        "text_redacted": True,
        "reasoning_generated_redacted": True,
    }
    path.write_text(
        json.dumps({**base, "reasoning_text": "testo generato"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reasoning"):
        package._assert_redacted(path)
    path.write_text(
        json.dumps({**base, "reasoning_text": None}) + "\n",
        encoding="utf-8",
    )
    package._assert_redacted(path)


def test_reasoning_metrics_separate_pro_ctcae_and_nonclassifiable():
    analysis = _script("analyze_reasoning_classification.py")
    rows = [
        {
            "correct": True,
            "system_correct": True,
            "pro_correct": True,
            "ctcae_correct": None,
            "nonclassifiable_correct": None,
            "explicit_nonclassifiable": False,
        },
        {
            "correct": False,
            "system_correct": True,
            "pro_correct": None,
            "ctcae_correct": False,
            "nonclassifiable_correct": None,
            "explicit_nonclassifiable": False,
        },
        {
            "correct": True,
            "system_correct": True,
            "pro_correct": None,
            "ctcae_correct": None,
            "nonclassifiable_correct": True,
            "explicit_nonclassifiable": True,
        },
    ]
    metrics = analysis._metrics(rows)
    assert metrics["strict_joint_accuracy"] == pytest.approx(2 / 3)
    assert metrics["pro_item_accuracy"] == 1.0
    assert metrics["ctcae_item_accuracy"] == 0.0
    assert metrics["nonclassifiable_accuracy"] == 1.0
    assert metrics["explicit_nonclassifiable_rate"] == pytest.approx(1 / 3)


def test_native_reasoning_is_reported_separately_from_portable_deliberation():
    analysis_module = _script("analyze_reasoning_classification.py")
    rows = []
    for index, direct_correct in enumerate((False, True)):
        for mode, correct, choice in (
            ("direct", direct_correct, f"direct-{index}"),
            ("deliberative", True, f"deliberative-{index}"),
            ("native_reasoning", True, f"native-{index}"),
        ):
            rows.append(
                {
                    "model": "qwen3-6-27b",
                    "record_id": f"record-{index}",
                    "source_id": f"source-{index}",
                    "role": "oncologo",
                    "reasoning_mode": mode,
                    "correct": correct,
                    "system_correct": correct,
                    "pro_correct": correct,
                    "ctcae_correct": None,
                    "nonclassifiable_correct": None,
                    "explicit_nonclassifiable": False,
                    "model_choice_id": choice,
                }
            )
    config = {
        "design": {
            "roles": ["oncologo"],
            "reasoning_modes": ["direct", "deliberative"],
        },
        "models": {"primary": ["Qwen/Qwen3.6-27B"], "secondary": []},
        "native_reasoning": {
            "enabled": True,
            "mode": "native_reasoning",
            "models": ["Qwen/Qwen3.6-27B"],
        },
        "analysis": {"primary_role": "oncologo", "bootstrap_draws": 50, "seed": 3},
        "model_pairs": {"primary": []},
    }
    report = analysis_module.analyse(rows, config)
    assert set(report["pooled_primary_role"]) == {"direct", "deliberative"}
    native = report["native_reasoning_exploratory"]["qwen3-6-27b"]
    assert native["metrics"]["strict_joint_accuracy"] == 1.0
    assert native["strict_accuracy_delta_native_minus_direct"]["estimate"] == 0.5


def test_real_runner_confines_and_clears_ephemeral_model_cache(tmp_path, monkeypatch):
    runner = _script("run_real_study.py")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    root = tmp_path / "model-cache"

    cache, env = runner._model_environment(root, "example/Model-8B")
    assert cache.parent == root.resolve()
    assert env["HF_HOME"] == str(cache)
    assert env["HF_HUB_CACHE"] == str(cache / "hub")
    assert env["HF_XET_CACHE"] == str(cache / "xet")
    assert env["HF_TOKEN"] == "test-token"

    (cache / "sentinel").write_text("temporary", encoding="utf-8")
    runner._safe_remove_model_cache(cache, root)
    assert root.is_dir()
    assert not cache.exists()
    with pytest.raises(ValueError, match="unsafe"):
        runner._safe_remove_model_cache(root, root)
    with pytest.raises(ValueError, match="unsafe"):
        runner._safe_remove_model_cache(tmp_path / "outside", root)


def test_complete_runner_uses_one_confined_cache_per_model(tmp_path, monkeypatch):
    runner = _script("run_complete_colab_study.py")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    root = tmp_path / "complete-cache"

    cache, env = runner._model_environment(root, "example/Model-8B")
    assert cache.parent == root.resolve()
    assert env["HF_HOME"] == str(cache)
    assert env["HF_HUB_CACHE"] == str(cache / "hub")
    assert env["HF_XET_CACHE"] == str(cache / "xet")
    assert env["HF_ASSETS_CACHE"] == str(cache / "assets")
    assert env["TRANSFORMERS_CACHE"] == str(cache / "transformers")
    assert env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"
    assert env["HF_XET_SHARD_CACHE_SIZE_LIMIT"] == "1000000000"

    (cache / "sentinel").write_text("temporary", encoding="utf-8")
    runner._remove_model_cache(cache, root)
    assert root.is_dir()
    assert not cache.exists()
    with pytest.raises(ValueError, match="unsafe"):
        runner._remove_model_cache(root, root)


def test_complete_runner_cohorts_are_aligned():
    runner = _script("run_complete_colab_study.py")
    primary = runner._configured_model_sets("primary")
    complete = runner._configured_model_sets("all")
    assert len(primary["controlled"]) == 8
    assert primary["controlled"] == primary["real"]
    assert len(primary["reasoning"]) == 9
    assert "FreedomIntelligence/Apollo2-7B" in primary["reasoning"]
    assert "swiss-ai/Apertus-8B-Instruct-2509" in primary["controlled"]
    assert "EPFLiGHT/Apertus-8B-MeditronFO" in primary["reasoning"]
    assert all("Apertus-70B" not in model for models in primary.values() for model in models)
    assert "mistralai/Ministral-8B-Instruct-2410" not in primary["reasoning"]
    assert len(complete["real"]) == 9
    assert len(complete["reasoning"]) == 10


def test_blackwell_runtime_uses_plain_bf16_for_every_model():
    runner = _script("run_complete_colab_study.py")
    import yaml

    profile = yaml.safe_load(
        (_ROOT / "configs/runtime_blackwell_96gb.yaml").read_text(encoding="utf-8")
    )
    base = runner._runtime_for_model(profile, "swiss-ai/Apertus-8B-Instruct-2509")
    medical = runner._runtime_for_model(profile, "EPFLiGHT/Apertus-8B-MeditronFO")
    apollo = runner._runtime_for_model(profile, "FreedomIntelligence/Apollo2-7B")
    assert base == medical == apollo
    assert base["quantization"] is None
    assert base["dtype"] == "bfloat16"
    assert base["minimum_free_disk_gb"] == 75.0


def test_reasoning_modes_add_native_thinking_only_for_qwen():
    runner = _script("analyze_reasoning_classification.py")
    import yaml

    config = yaml.safe_load(
        (_ROOT / "configs/study_esmo_2026_reasoning.yaml").read_text(encoding="utf-8")
    )
    assert runner._reasoning_modes_for_model(config, "Qwen/Qwen3.6-27B") == [
        "direct",
        "deliberative",
        "native_reasoning",
    ]
    assert runner._reasoning_modes_for_model(
        config, "google/gemma-3-27b-it"
    ) == ["direct", "deliberative"]


def test_qwen36_uses_memory_safe_candidate_chunk(monkeypatch):
    monkeypatch.syspath_prepend(str(_ROOT / "scripts"))
    runner = _script("run_reasoning_study.py")
    import yaml

    config = yaml.safe_load(
        (_ROOT / "configs/study_esmo_2026_reasoning.yaml").read_text(encoding="utf-8")
    )
    assert runner._candidate_chunk_for_model(config, "Qwen/Qwen3.6-27B", 32) == 8
    assert runner._candidate_chunk_for_model(config, "FreedomIntelligence/Apollo2-7B", 32) == 32


def test_item_audit_adds_reasoning_direct_and_deliberative_payloads(
    tmp_path, monkeypatch
):
    exporter = _script("export_item_audit.py")
    dataset = tmp_path / "clinical_real.jsonl"
    source = {
        "record_id": "real-0001",
        "source_id": "source-1",
        "source_row": 2,
        "text": "nausea",
        "n_words": 1,
        "annotation_source": "PRO-CTCAE",
        "source_item": "Nausea",
        "grade": 2,
        "gold_class": "term",
        "gold_pro_id": "PRO_001",
        "gold_pro_status": "exact",
        "gold_pro_term": "Nausea",
        "gold_ctcae_term": None,
    }
    dataset.write_text(json.dumps(source) + "\n", encoding="utf-8")
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()

    v1_dir = tmp_path / "v1"
    reasoning_dir = tmp_path / "reasoning"
    outdir = tmp_path / "audit"
    v1_dir.mkdir()
    reasoning_dir.mkdir()
    v1_result = {
        **{
            key: source[key]
            for key in (
                "record_id",
                "source_id",
                "source_row",
                "source_item",
                "annotation_source",
                "grade",
                "n_words",
                "gold_class",
                "gold_pro_id",
                "gold_pro_status",
            )
        },
        "role": "oncologo",
        "arm": "intact",
        "correct": True,
        "model_top1_id": "PRO_001",
        "model_top1_term": "Nausea",
    }
    (v1_dir / "model__rows.jsonl").write_text(
        json.dumps(v1_result) + "\n", encoding="utf-8"
    )
    (v1_dir / "model__meta.json").write_text(
        json.dumps({"model_id": "example/model", "dataset_sha256": dataset_sha}),
        encoding="utf-8",
    )

    reasoning_rows = []
    for mode in ("direct", "deliberative"):
        reasoning_rows.append(
            {
                **{
                    key: source[key]
                    for key in (
                        "record_id",
                        "source_id",
                        "source_row",
                        "source_item",
                        "annotation_source",
                        "grade",
                        "n_words",
                        "gold_pro_id",
                        "gold_ctcae_term",
                    )
                },
                "gold_choice_id": "PRO::PRO_001",
                "gold_system": "PRO-CTCAE",
                "gold_item": "Nausea",
                "role": "oncologo",
                "reasoning_mode": mode,
                "model_choice_id": "PRO::PRO_001",
                "model_choice_system": "PRO-CTCAE",
                "model_choice_item": "Nausea",
                "correct": True,
                "system_correct": True,
                "explicit_nonclassifiable": False,
                "top5": [],
                "reasoning_n_tokens": 0 if mode == "direct" else 20,
                "reasoning_sha256_16": None if mode == "direct" else "abc123",
            }
        )
    (reasoning_dir / "model__rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in reasoning_rows),
        encoding="utf-8",
    )
    (reasoning_dir / "model__meta.json").write_text(
        json.dumps({"model_id": "example/model", "dataset_sha256": dataset_sha}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_item_audit.py",
            "--results-dir",
            str(v1_dir),
            "--reasoning-results-dir",
            str(reasoning_dir),
            "--dataset",
            str(dataset),
            "--outdir",
            str(outdir),
        ],
    )
    assert exporter.main() == 0
    direct_and_deliberative = [
        json.loads(line)
        for line in (outdir / "reasoning_audit_long.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["reasoning_mode"] for row in direct_and_deliberative} == {
        "direct",
        "deliberative",
    }
    wide = json.loads(
        (outdir / "reasoning_audit_wide.jsonl").read_text(encoding="utf-8")
    )
    assert set(wide["models"]["example/model"]) == {"direct", "deliberative"}
