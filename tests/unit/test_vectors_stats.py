"""Disentanglement math, steering ops, random vectors, statistics."""

from __future__ import annotations

import numpy as np

from oncoemotion.emotion_vectors import orthogonalize, cosine, random_vector
from oncoemotion.steering.spec import steer_add, ablate_projection, norm_scaled_alpha
from oncoemotion.statistics import bootstrap_ci, benjamini_hochberg


def test_orthogonalize_removes_confounder_component():
    rng = np.random.default_rng(0)
    d = 32
    C = rng.standard_normal((3, d))
    v = rng.standard_normal(d)
    v_perp = orthogonalize(v, C)
    # residual is orthogonal to every confounder
    for c in C:
        assert abs(float(np.dot(v_perp, c))) < 1e-8


def test_orthogonalize_none_is_identity():
    v = np.arange(5.0)
    assert np.allclose(orthogonalize(v, None), v)


def test_orthogonalize_vector_in_span_goes_to_zero():
    C = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v = np.array([2.0, -3.0, 0.0])  # lies in span(C)
    assert np.linalg.norm(orthogonalize(v, C)) < 1e-8


def test_random_vector_reproducible_and_normed():
    a = random_vector(64, seed=123, norm=2.0)
    b = random_vector(64, seed=123, norm=2.0)
    c = random_vector(64, seed=999, norm=2.0)
    assert np.allclose(a, b)               # reproducible with seed
    assert not np.allclose(a, c)           # different seed differs
    assert abs(np.linalg.norm(a) - 2.0) < 1e-9


def test_steer_add_does_not_mutate_input():
    h = np.ones(4)
    v = np.array([1.0, 0.0, 0.0, 0.0])
    out = steer_add(h, v, 0.5)
    assert np.allclose(h, np.ones(4))      # input unchanged (no permanent edit)
    assert np.allclose(out, [1.5, 1.0, 1.0, 1.0])


def test_ablate_projection_removes_direction():
    h = np.array([[3.0, 4.0, 0.0]])
    v = np.array([1.0, 0.0, 0.0])
    out = ablate_projection(h, v)
    assert abs(float((out @ v)[0])) < 1e-9   # component along v removed
    assert np.allclose(h, [[3.0, 4.0, 0.0]])  # input unchanged
    # also verify the 1-D path
    out1 = ablate_projection(np.array([3.0, 4.0, 0.0]), v)
    assert abs(float(out1 @ v)) < 1e-9


def test_norm_scaled_alpha():
    assert norm_scaled_alpha(0.05, residual_norm=10.0, ref_norm=1.0) == 0.5


def test_cosine():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_bootstrap_ci_deterministic():
    data = list(range(100))
    p1 = bootstrap_ci(data, seed=7, n_boot=500)
    p2 = bootstrap_ci(data, seed=7, n_boot=500)
    assert p1 == p2                        # deterministic given seed
    point, lo, hi = p1
    assert lo <= point <= hi


def test_bootstrap_ci_stratified_runs():
    data = [1, 2, 3, 4, 5, 6]
    strata = ["a", "a", "a", "b", "b", "b"]
    point, lo, hi = bootstrap_ci(data, seed=1, n_boot=200, strata=strata)
    assert lo <= point <= hi


def test_benjamini_hochberg():
    pvals = [0.001, 0.2, 0.03, 0.9]
    rej = benjamini_hochberg(pvals, alpha=0.05)
    assert rej[0] and not rej[3]
