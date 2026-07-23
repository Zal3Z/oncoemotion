"""Phase 3 unit tests: decision prompt, measure dataset, projection/z-score."""

from __future__ import annotations

import numpy as np

from oncoemotion.clinical.prompt import build_decision_prompt, TEACHER_PREFIX, NEUTRAL_FILLER
from oncoemotion.clinical.measure import project_scores, zscore
from oncoemotion.clinical.measure_dataset import build_measure_items


def test_prompt_ends_with_teacher_prefix():
    p = build_decision_prompt("ho la nausea")
    assert p.endswith(TEACHER_PREFIX)          # last token = point E
    assert "ho la nausea" in p


def test_prompt_inserts_neutral_filler():
    p = build_decision_prompt("ho la nausea", neutral_filler=NEUTRAL_FILLER)
    assert NEUTRAL_FILLER in p
    assert p.endswith(TEACHER_PREFIX)
    # filler sits before the decision prefix
    assert p.index(NEUTRAL_FILLER) < p.index(TEACHER_PREFIX)


def test_measure_items_structure():
    items = build_measure_items()
    ids = [i.item_id for i in items]
    assert len(ids) == len(set(ids))           # unique ids
    assert any(i.is_neutral for i in items)     # baseline present
    assert any(i.group.startswith("gradient:") for i in items)
    # gradients are ordered by step within group
    grad = [i for i in items if i.group == "gradient:pain"]
    assert [g.step for g in grad] == sorted(g.step for g in grad)


def test_projection_and_zscore():
    H, L1 = 4, 3
    hidden = np.zeros((L1, H))
    hidden[1] = np.array([2.0, 0.0, 0.0, 0.0])   # point-E hidden at layer 1
    vectors = {"afraid_alarmed": np.zeros((L1, H))}
    vectors["afraid_alarmed"][1] = np.array([1.0, 0.0, 0.0, 0.0])  # direction at layer 1
    scores = project_scores(hidden, vectors, {"afraid_alarmed": 1})
    assert abs(scores["afraid_alarmed"] - 2.0) < 1e-9   # projection = 2
    z = zscore(scores, {"afraid_alarmed": 0.0}, {"afraid_alarmed": 1.0})
    assert abs(z["afraid_alarmed"] - 2.0) < 1e-9
