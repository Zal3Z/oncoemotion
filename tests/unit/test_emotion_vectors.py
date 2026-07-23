"""Phase 2 unit tests: dataset, vector construction, probing, extraction."""

from __future__ import annotations

import numpy as np
import pytest

from oncoemotion.emotion_vectors.dataset import build_dataset
from oncoemotion.emotion_vectors.build import build_layer_vector, build_all_layers, METHODS
from oncoemotion.emotion_vectors.vectors import cosine
from oncoemotion.probing.probe import evaluate_direction, projection_scores


def test_dataset_deterministic_and_labeled():
    a = build_dataset(seed=7)
    b = build_dataset(seed=7)
    assert [(e.concept, e.text, e.label, e.split) for e in a] == \
           [(e.concept, e.text, e.label, e.split) for e in b]
    assert {e.label for e in a} == {0, 1}
    assert {e.split for e in a} == {"extraction", "validation", "test"}
    # both classes present
    assert any(e.label == 1 for e in a) and any(e.label == 0 for e in a)


def _separable(seed=0, d=16, n=30, sep=1.2):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(sep, 0.4, (n, d)), rng.normal(-sep, 0.4, (n, d))])
    y = np.array([1] * n + [0] * n)
    return X, y


@pytest.mark.parametrize("method", METHODS)
def test_methods_separate_classes(method):
    X, y = _separable()
    v = build_layer_vector(X, y, method, "c", 3)
    res = evaluate_direction(X, y, v.vector)
    assert res["auroc"] > 0.9
    assert v.original_norm > 0


def test_direction_oriented_toward_positive():
    X, y = _separable()
    v = build_layer_vector(X, y, "diff_of_means", "c", 3).vector
    # positive class should project higher than negative on the oriented direction
    s = projection_scores(X, v)
    assert s[y == 1].mean() > s[y == 0].mean()


def test_build_all_layers_shape():
    n, L1, H = 40, 5, 16
    acts = np.random.default_rng(1).normal(size=(n, L1, H))
    y = np.array([1] * 20 + [0] * 20)
    vecs = build_all_layers(acts, y, "c", "diff_of_means")
    assert len(vecs) == L1
    assert all(v.vector.shape == (H,) for v in vecs)
    assert [v.layer for v in vecs] == list(range(L1))


def test_cosine_of_identical_directions():
    v = np.array([0.3, -1.2, 4.0])
    assert abs(cosine(v, 2 * v) - 1.0) < 1e-9


def test_pooled_hidden_states_with_fake_adapter():
    torch = pytest.importorskip("torch")
    from oncoemotion.activations.extract import pooled_hidden_states

    H, L1 = 8, 4

    class FakeAdapter:
        def forward_capture(self, text):
            seq = 3 + len(text) % 3
            hs = tuple(torch.arange(seq * H, dtype=torch.float32).reshape(1, seq, H) for _ in range(L1))
            attn = torch.ones(1, seq, dtype=torch.long)
            return {"hidden_states": hs, "attention_mask": attn}

    acts = pooled_hidden_states(FakeAdapter(), ["a", "bb", "ccc"], pooling="mean")
    assert acts.shape == (3, L1, H)
    # last pooling path also works
    acts2 = pooled_hidden_states(FakeAdapter(), ["a", "bb"], pooling="last")
    assert acts2.shape == (2, L1, H)
