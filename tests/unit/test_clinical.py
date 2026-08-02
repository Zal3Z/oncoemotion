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


class _FakeTok:
    """Whitespace tokenizer: enough to exercise the padding logic without a model."""

    def __call__(self, text, add_special_tokens=False):
        class _Enc:
            input_ids = text.split()
        return _Enc()


def test_padded_role_spans_are_token_matched():
    from oncoemotion.clinical.prompt import ROLE_PERSONAS, build_padded_personas

    personas, counts = build_padded_personas(_FakeTok())
    assert set(personas) == set(ROLE_PERSONAS)
    # the whole point: everything after the system block must land at the same
    # absolute position in every condition
    assert max(counts.values()) - min(counts.values()) <= 2


def test_no_role_control_has_a_system_block():
    """'none' used to be the absence of a system message, which is a structurally
    different prompt rather than a control."""
    from oncoemotion.clinical.prompt import ROLE_PERSONAS, build_padded_personas

    assert ROLE_PERSONAS["none"] is None
    personas, counts = build_padded_personas(_FakeTok())
    assert personas["none"]
    assert counts["none"] >= max(counts.values()) - 2


def test_padded_personas_keep_their_identity():
    from oncoemotion.clinical.prompt import build_padded_personas

    personas, _ = build_padded_personas(_FakeTok())
    assert personas["oncologo"].startswith("Sei un oncologo")
    assert "oncologo" not in personas["none"]


def test_build_decision_messages_uses_the_padded_table():
    from oncoemotion.clinical.prompt import build_decision_messages, build_padded_personas

    personas, _ = build_padded_personas(_FakeTok())
    sys_none, _ = build_decision_messages("ho nausea", role="none", personas=personas)
    assert sys_none is not None
    sys_default, _ = build_decision_messages("ho nausea", role="none")
    assert sys_default is None      # unpadded default keeps the old behaviour
