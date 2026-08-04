from __future__ import annotations

import importlib.util
import json
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
