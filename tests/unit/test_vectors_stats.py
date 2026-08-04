"""Disentanglement math, steering ops, random vectors, statistics."""

from __future__ import annotations

import numpy as np

from oncoemotion.emotion_vectors import cosine, orthogonalize, random_vector
from oncoemotion.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    hierarchical_cluster_ci,
)
from oncoemotion.steering.spec import ablate_projection, norm_scaled_alpha, steer_add


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


def test_bh_adjusted_pvalues_are_monotone_and_bounded():
    from oncoemotion.statistics import bh_adjusted_pvalues

    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    adj = bh_adjusted_pvalues(p)
    assert len(adj) == len(p)
    assert all(0.0 <= a <= 1.0 for a in adj)
    # BH is a step-up procedure: adjusted values must not decrease with raw p
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranked = [adj[i] for i in order]
    assert ranked == sorted(ranked)
    assert bh_adjusted_pvalues([]).size == 0


def test_bh_adjusted_agrees_with_rejections():
    from oncoemotion.statistics import benjamini_hochberg, bh_adjusted_pvalues

    p = [0.001, 0.02, 0.3, 0.7]
    rej = benjamini_hochberg(p, alpha=0.05)
    adj = bh_adjusted_pvalues(p)
    assert list(rej) == [a <= 0.05 for a in adj]


def test_hierarchical_cluster_ci_keeps_models_equally_weighted():
    values = {
        "small": {"p1": [0.0], "p2": [0.0]},
        "large": {f"p{i}": [1.0, 1.0] for i in range(20)},
    }
    point, lo, hi = hierarchical_cluster_ci(values, n_boot=400, seed=11)
    # Macro-model estimand: 0 and 1 receive equal weight even though the second
    # model contributed ten times as many clinical pairs and two framings each.
    assert point == 0.5
    assert lo <= point <= hi


def test_hierarchical_cluster_ci_is_deterministic_and_pair_clustered():
    values = {
        "m1": {"p1": [0.0, 1.0], "p2": [1.0, 1.0]},
        "m2": {"p1": [0.0, 0.0], "p2": [0.0, 1.0]},
    }
    a = hierarchical_cluster_ci(values, n_boot=250, seed=19)
    b = hierarchical_cluster_ci(values, n_boot=250, seed=19)
    assert a == b
    assert a[1] <= a[0] <= a[2]
