"""Invariant ("theorem") suite for libdmkde, checked with property-based testing.

Each test states a mathematical guarantee of DMKDE and lets Hypothesis search for a
counterexample. A failing test is a counterexample: a concrete input that breaks the claim.

Moved from audit/test_invariants.py in 0.1.1. Imports the installed dmkde; for an
in-place build run ``PYTHONPATH=python pytest tests``.
"""
from __future__ import annotations

import pickle

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import dmkde
from dmkde import _core

SETTINGS = settings(max_examples=300, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])

dims = st.integers(1, 6)
feat = st.sampled_from([2, 4, 8, 32])
sigmas = st.floats(0.05, 50.0)
seeds = st.integers(0, 2**63 - 1)
finite = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)


@st.composite
def dataset(draw, n_min=1, n_max=40):
    d = draw(dims)
    n = draw(st.integers(n_min, n_max))
    X = draw(arrays(np.float64, (n, d), elements=finite))
    return X


def model(X, D, sigma, seed):
    m = _core.DMKDE(X.shape[1], D, sigma, seed)
    m.fit(np.ascontiguousarray(X))
    return m


# Theorem 1: the RFF embedding is a unit vector (sum of sin^2+cos^2 scaled by 2/D).
@SETTINGS
@given(X=dataset(), D=feat, sigma=sigmas, seed=seeds)
def test_T1_embedding_is_unit_norm(X, D, sigma, seed):
    m = model(X, D, sigma, seed)
    for x in X[:5]:
        assert abs(np.linalg.norm(m.transform(x)) - 1.0) < 1e-9


# Theorem 2: rho is a density matrix: symmetric, PSD, trace 1.
@SETTINGS
@given(X=dataset(), D=feat, sigma=sigmas, seed=seeds)
def test_T2_rho_is_a_density_matrix(X, D, sigma, seed):
    m = model(X, D, sigma, seed)
    R = m.rho_matrix()
    assert np.allclose(R, R.T, atol=1e-12)
    assert abs(np.trace(R) - 1.0) < 1e-9
    assert np.linalg.eigvalsh(R).min() > -1e-10


# Theorem 3: Born-rule score is a probability: 0 <= s <= 1.
@SETTINGS
@given(X=dataset(), Q=arrays(np.float64, (5, 6), elements=finite), D=feat, sigma=sigmas, seed=seeds)
def test_T3_score_in_unit_interval(X, Q, D, sigma, seed):
    m = model(X, D, sigma, seed)
    s = m.score_batch(np.ascontiguousarray(Q[:, : X.shape[1]]))
    assert np.all(s >= -1e-12) and np.all(s <= 1 + 1e-9)


# Theorem 4: the C++ score equals an independent numpy reference (1/N) sum (phi(q).phi(x_i))^2.
@SETTINGS
@given(X=dataset(), D=feat, sigma=sigmas, seed=seeds)
def test_T4_matches_independent_reference(X, D, sigma, seed):
    m = model(X, D, sigma, seed)
    P = np.array([m.transform(x) for x in X])
    q = X[0] * 0.5
    ref = np.mean((P @ m.transform(q)) ** 2)
    assert abs(m.score_batch(q[None, :])[0] - ref) < 1e-9


# Theorem 5: fitting does not depend on row order.
@SETTINGS
@given(X=dataset(n_min=2), D=feat, sigma=sigmas, seed=seeds, perm_seed=st.integers(0, 10**6))
def test_T5_fit_is_permutation_invariant(X, D, sigma, seed, perm_seed):
    a = model(X, D, sigma, seed).rho_matrix()
    b = model(X[np.random.default_rng(perm_seed).permutation(len(X))], D, sigma, seed).rho_matrix()
    assert np.allclose(a, b, atol=1e-12)


# Theorem 6: the streaming update keeps rho a density matrix; alpha=1 forgets everything else.
@SETTINGS
@given(X=dataset(n_min=2), D=feat, sigma=sigmas, seed=seeds, alpha=st.floats(1e-6, 1.0))
def test_T6_streaming_update_preserves_density_matrix(X, D, sigma, seed, alpha):
    m = model(X[:1], D, sigma, seed)
    for x in X[1:]:
        m.update(np.ascontiguousarray(x), alpha)
    R = m.rho_matrix()
    assert abs(np.trace(R) - 1.0) < 1e-9
    assert np.linalg.eigvalsh(R).min() > -1e-10
    m.update(np.ascontiguousarray(X[0]), 1.0)
    assert abs(m.score_batch(X[:1])[0] - 1.0) < 1e-9


# Theorem 7: roc_auc equals scikit-learn's on any finite scores, including ties.
@SETTINGS
@given(a=arrays(np.float64, st.integers(1, 30), elements=st.sampled_from([0.0, 0.5, 1.0, 2.0, 3.5])),
       b=arrays(np.float64, st.integers(1, 30), elements=st.sampled_from([0.0, 0.5, 1.0, 2.0, 3.5])))
def test_T7_roc_auc_matches_sklearn(a, b):
    from sklearn.metrics import roc_auc_score
    ref = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    assert abs(dmkde.roc_auc(a, b) - ref) < 1e-12


# Theorem 8: the Gaussian kernel is shift invariant, so scores must not change if every
# point (training and query) is translated by the same vector c.
@settings(max_examples=60, deadline=None)
@given(X=dataset(n_min=5), c=st.floats(1e3, 1e9), D=st.sampled_from([64, 256]), seed=seeds)
def test_T8_shift_invariance(X, c, D, seed):
    sigma = 1.0
    X = np.clip(X, -3, 3)
    q = X[:3] * 0.9
    base = model(X, D, sigma, seed).score_batch(np.ascontiguousarray(q))
    shifted = model(X + c, D, sigma, seed).score_batch(np.ascontiguousarray(q + c))
    assert np.allclose(base, shifted, atol=1e-6), (c, base, shifted)


# Theorem 9: RFF approximates the documented kernel exp(-||x-y||^2 / 2 sigma^2) within O(1/sqrt(D)).
@settings(max_examples=40, deadline=None)
@given(seed=seeds, sigma=st.floats(0.5, 5.0))
def test_T9_rff_approximates_documented_kernel(seed, sigma):
    rng = np.random.default_rng(seed % 2**32)
    X = rng.normal(size=(30, 4))
    m = model(X, 8192, sigma, seed)
    P = np.array([m.transform(x) for x in X])
    K = P @ P.T
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    assert np.abs(K - np.exp(-d2 / (2 * sigma**2))).max() < 6 / np.sqrt(8192)


# Theorem 10 (0.1.1): pickling is lossless — the restored estimator gives bit-identical
# scores, decisions and rho, including after streaming updates.
@settings(max_examples=60, deadline=None)
@given(X=dataset(n_min=2), D=feat, sigma=sigmas, seed=seeds, k=st.integers(0, 5))
def test_T10_pickle_roundtrip_is_exact(X, D, sigma, seed, k):
    m = dmkde.DMKDE(feature_dim=D, sigma=sigma, random_state=seed).fit(X)
    if k:
        m.partial_fit(X[:k], alpha=0.1)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.score_samples(X), m.score_samples(X))
    np.testing.assert_array_equal(m2.decision_function(X), m.decision_function(X))
    np.testing.assert_array_equal(m2._impl.rho_matrix(), m._impl.rho_matrix())


# Theorem 11 (0.1.1): sklearn outlier contract — decision_function = score_samples - offset_
# and predict == where(decision_function >= 0, 1, -1).
@settings(max_examples=60, deadline=None)
@given(X=dataset(n_min=2), D=feat, sigma=sigmas, seed=seeds, c=st.floats(0.01, 0.5))
def test_T11_predict_is_sign_of_decision_function(X, D, sigma, seed, c):
    m = dmkde.DMKDE(feature_dim=D, sigma=sigma, random_state=seed, contamination=c).fit(X)
    df = m.decision_function(X)
    np.testing.assert_allclose(df, m.score_samples(X) - m.offset_)
    np.testing.assert_array_equal(m.predict(X), np.where(df >= 0, 1, -1))


# Theorem 12 (0.1.1): Mahalanobis distance is invariant to rescaling the data.
@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), c=st.floats(1e-6, 1e6), d=st.integers(1, 6))
def test_T12_mahalanobis_scale_invariance(seed, c, d):
    r = np.random.default_rng(seed)
    X = r.normal(size=(50, d)) @ r.normal(size=(d, d))
    Q = r.normal(size=(10, d))
    if np.linalg.matrix_rank(X - X.mean(0)) < d:
        return
    base = dmkde.Mahalanobis().fit(X).score_samples(Q)
    scaled = dmkde.Mahalanobis().fit(X * c).score_samples(Q * c)
    np.testing.assert_allclose(scaled, base, rtol=1e-6, atol=1e-9)
