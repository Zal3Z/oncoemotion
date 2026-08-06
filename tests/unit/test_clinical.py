"""Phase 3 unit tests: decision prompt, measure dataset, projection/z-score."""

from __future__ import annotations

import numpy as np

from oncoemotion.clinical.measure import project_scores, zscore
from oncoemotion.clinical.measure_dataset import build_measure_items
from oncoemotion.clinical.prompt import NEUTRAL_FILLER, TEACHER_PREFIX, build_decision_prompt


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


def test_generated_outcomes_keep_abstention_and_failure_separate():
    from oncoemotion.clinical.classify import classify_generated_term

    def matcher(text):
        return ("PRO_002", "Difficulty swallowing", 0.99) if text == "Disfagia" \
            else (None, None, 0.2)

    assert classify_generated_term("Disfagia", matcher) == ("PRO_002", "mapped", 0.99)
    assert classify_generated_term("nessun evento avverso", matcher)[1] == "abstained"
    assert classify_generated_term("CTCAE_5.0_Grade_", matcher)[1] == "non_answer"
    assert classify_generated_term("parola inventata", matcher)[1] == "unmapped"


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


def test_joint_prompt_offers_both_ontologies_and_explicit_abstention():
    from oncoemotion.clinical.prompt import build_decision_messages

    _, user = build_decision_messages(
        "ho nausea",
        role="none",
        decision_space="joint",
    )
    assert "PRO-CTCAE | <item>" in user
    assert "CTCAE | <item>" in user
    assert "NON_CLASSIFICABILE" in user


def test_joint_gold_keeps_ctcae_separate_from_nonclassifiable():
    from oncoemotion.clinical.joint import gold_joint_choice

    pro = gold_joint_choice(
        {
            "annotation_source": "PRO-CTCAE",
            "gold_pro_id": "PRO_048",
            "gold_pro_term": "General pain",
        }
    )
    ctcae = gold_joint_choice(
        {"annotation_source": "CTCAE v5", "gold_ctcae_term": "Fever"}
    )
    unknown = gold_joint_choice({"annotation_source": "Non associabile"})
    assert pro.choice_id == "PRO::PRO_048"
    assert ctcae.choice_id == "CTCAE::Fever"
    assert unknown.choice_id == "NON_CLASSIFICABILE"
