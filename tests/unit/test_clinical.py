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
    """Every span that HAS a system block must be the same length: what follows it
    then sits at the same absolute position in every condition."""
    from oncoemotion.clinical.prompt import build_padded_personas

    personas, counts = build_padded_personas(_FakeTok())
    padded = [n for k, n in counts.items() if personas[k] is not None]
    assert max(padded) - min(padded) <= 2


def test_the_three_controls_are_distinct():
    """The first version had one control that silently named the task, so
    'role vs no role' actually measured 'identity framing vs task framing' -- and
    the control won. They are separate arms now and must stay separate."""
    from oncoemotion.clinical.prompt import build_padded_personas

    p, _ = build_padded_personas(_FakeTok())
    assert p["none"] is None                     # literally no system block
    assert p["none_filler"]                      # padded, no identity, no task
    assert p["none_task"]                        # padded, names the task
    assert p["none_filler"] != p["none_task"]


def test_the_filler_control_names_neither_task_nor_identity():
    from oncoemotion.clinical.prompt import build_padded_personas

    p, _ = build_padded_personas(_FakeTok())
    low = p["none_filler"].lower()
    for banned in ("codifica", "pro-ctcae", "paziente", "sintomo", "termine", "sei un"):
        assert banned not in low, f"il controllo nomina {banned!r}: non e' inerte"
    assert "codifica" in p["none_task"].lower()   # questo invece il compito lo nomina


def test_padded_personas_keep_their_identity():
    from oncoemotion.clinical.prompt import build_padded_personas

    personas, _ = build_padded_personas(_FakeTok())
    assert personas["oncologo"].startswith("Sei un oncologo")
    assert "oncologo" not in personas["none_filler"]


def test_build_decision_messages_uses_the_padded_table():
    from oncoemotion.clinical.prompt import build_decision_messages, build_padded_personas

    personas, _ = build_padded_personas(_FakeTok())
    sys_ctrl, _ = build_decision_messages("ho nausea", role="none_filler", personas=personas)
    assert sys_ctrl is not None
    sys_default, _ = build_decision_messages("ho nausea", role="none")
    assert sys_default is None      # unpadded default keeps the old behaviour
