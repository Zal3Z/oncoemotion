"""Design constraints on the paired clinical item set.

These are the guardrails that keep the role x framing endpoint interpretable. The
previous item set failed all of them: the emotional pole was 2.5x longer than the
neutral one (median 10 words vs 4) and carried commas in 40% of items vs 1%, so an
accuracy drop under emotional framing could not be told apart from an accuracy drop
under longer, more punctuated text.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_GEN = _ROOT / "scripts" / "generate_labeled_clinical.py"
_DATA = _ROOT / "data" / "synthetic" / "clinical_labeled.jsonl"


def _generator():
    spec = importlib.util.spec_from_file_location("_gen_clinical", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _generator()


@pytest.fixture(scope="module")
def recs(gen):
    return gen._records(gen._seeds())


def test_authored_seeds_satisfy_every_design_constraint(gen, recs):
    """The single source of truth: the generator's own check must come back clean."""
    assert gen._check(recs) == []


def test_poles_differ_only_in_the_qualifier(gen):
    """Neutral and emotional text must be identical outside the {q} slot."""
    for pid, _term, tpl, qn, qe in gen.TERM_SEEDS:
        neu, emo = gen._render(tpl, qn), gen._render(tpl, qe)
        assert neu != emo, f"{pid}: the two framings are identical"
        # removing the filler from each must leave the same skeleton
        assert neu.replace(qn, "@") == emo.replace(qe, "@"), (
            f"{pid}: framings differ outside the qualifier slot:\n  {neu}\n  {emo}")


def test_qualifier_pairs_are_length_matched(gen):
    for pid, _term, _tpl, qn, qe in gen.TERM_SEEDS:
        dw = abs(len(qn.split()) - len(qe.split()))
        assert dw <= 1, f"{pid}: qualifiers differ by {dw} words ({qn!r} vs {qe!r})"


def test_no_exclamation_marks_anywhere(recs):
    """The reference corpus of 1194 real responses contains none."""
    offenders = [r["record_id"] for r in recs if "!" in r["text"]]
    assert offenders == []


def test_enough_paired_term_items_for_the_primary_endpoint(recs):
    """With 35 pairs the role effect was ~1.1 items, below the resolution floor."""
    pairs = {r["pair_id"] for r in recs if r["gold_class"] == "term"}
    assert len(pairs) >= 100


def test_written_dataset_matches_the_generator(gen, recs):
    """Guards against a stale committed snapshot."""
    if not _DATA.exists():
        pytest.skip("dataset not generated in this checkout")
    on_disk = [json.loads(line) for line in _DATA.read_text(encoding="utf-8").splitlines()]
    assert len(on_disk) == len(recs)
    assert [r["text"] for r in on_disk] == [r["text"] for r in recs]


def test_deduplication_fields_present(recs):
    """Real ingested text has 25% exact duplicates; the schema must already carry
    the keys needed to collapse them, so adding real data is not a migration."""
    for r in recs:
        assert "source_id" in r and "assessment_id" in r
