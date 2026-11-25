"""Regression tests for the libdmkde 0.1.1 safety release.

One test per in-scope finding from the two audits (``audit/test_findings.py``
F*, ``review2/p*.py``). Each test asserts the FIXED behaviour, so every test
here fails against dmkde 0.1.0 and passes on 0.1.1.

Cases that could crash the interpreter on a broken build (out-of-bounds
reads, NULL dereferences, huge allocations) run in a child process via
``run_isolated`` so a crash is reported as a failure instead of killing the
whole test session.

The tests import the installed ``dmkde``. To run them against an in-place
build use ``PYTHONPATH=python pytest tests``.
"""
from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

import dmkde
from dmkde import _core

ROOT = Path(__file__).resolve().parents[1]
PKG_PARENT = str(Path(dmkde.__file__).resolve().parents[1])

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5))


def fitted(**kw) -> dmkde.DMKDE:
    return dmkde.DMKDE(feature_dim=64, sigma=1.0, **kw).fit(X)


def run_isolated(code: str, limit_memory: bool = False) -> str:
    """Run ``code`` in a fresh interpreter that imports the same dmkde.

    The child must print ``OK`` on success. Returns its combined output;
    fails the test on a non-zero exit (e.g. a segfault) or a missing ``OK``.
    """
    prelude = "import numpy as np, dmkde\nfrom dmkde import _core\n"
    if (limit_memory and sys.platform.startswith("linux")
            and "asan" not in os.environ.get("LD_PRELOAD", "")):
        prelude += ("import resource\n"
                    "resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)],
                       capture_output=True, text=True, env=env, timeout=300)
    out = p.stdout + p.stderr
    assert p.returncode == 0, f"child exited {p.returncode}:\n{out}"
    assert "OK" in p.stdout, out
    return out


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not available (not a source checkout)")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Memory safety and shapes  (audit F1, F1b, F12, F13; review2 p01, p15)
# ---------------------------------------------------------------------------
EXPECT_VALUE_ERROR = """
def expect(fn, pattern):
    try:
        fn()
    except ValueError as e:
        assert pattern in str(e), str(e)
        return
    raise AssertionError("no ValueError")
rng = np.random.default_rng(0)
X = rng.normal(size=(50, 8))
"""


def test_F1_p01_score_samples_fewer_columns_rejected():
    run_isolated(EXPECT_VALUE_ERROR + """
m = dmkde.DMKDE(feature_dim=64).fit(X)
expect(lambda: m.score_samples(rng.normal(size=(3, 2))), "features")
print("OK")""")


def test_F1b_p01_score_samples_extra_columns_rejected():
    m = fitted()
    with pytest.raises(ValueError, match="features"):
        m.score_samples(np.hstack([X[:10], rng.normal(size=(10, 3))]))


def test_p01_zero_column_query_rejected():
    run_isolated(EXPECT_VALUE_ERROR + """
m = dmkde.DMKDE(feature_dim=64).fit(X)
expect(lambda: m.score_samples(np.empty((2, 0))), "feature")
c = _core.DMKDE(8, 64, 1.0, 1); c.fit(X)
expect(lambda: c.score_batch(np.empty((2, 0))), "features")
print("OK")""")


def test_F13_zero_feature_fit_rejected():
    with pytest.raises(ValueError, match="0 feature"):
        dmkde.DMKDE(feature_dim=32).fit(np.empty((50, 0)))


def test_p01_core_score_width_checked():
    run_isolated(EXPECT_VALUE_ERROR + """
c = _core.DMKDE(8, 64, 1.0, 1); c.fit(X)
expect(lambda: c.score(np.zeros(1)), "features")
expect(lambda: c.score(np.zeros(20)), "features")
print("OK")""")


def test_p01_core_score_batch_width_checked():
    run_isolated(EXPECT_VALUE_ERROR + """
c = _core.DMKDE(8, 64, 1.0, 1); c.fit(X)
expect(lambda: c.score_batch(np.zeros((3, 2))), "features")
expect(lambda: c.score_batch(np.zeros((3, 9))), "features")
print("OK")""")


def test_p01_core_transform_width_checked():
    run_isolated(EXPECT_VALUE_ERROR + """
c = _core.DMKDE(8, 64, 1.0, 1); c.fit(X)
expect(lambda: c.transform(np.zeros(1)), "features")
print("OK")""")


def test_p01_mahalanobis_width_checked():
    run_isolated(EXPECT_VALUE_ERROR + """
mh = dmkde.Mahalanobis().fit(X)
expect(lambda: mh.score_samples(rng.normal(size=(3, 2))), "features")
c = _core.Mahalanobis(); c.fit(X)
expect(lambda: c.score(np.zeros(2)), "features")
expect(lambda: c.score_batch(np.zeros((3, 2))), "features")
print("OK")""")


def test_p01_mahalanobis_zero_features_rejected():
    with pytest.raises(ValueError):
        _core.Mahalanobis().fit(np.empty((5, 0)))


def test_p01_mahalanobis_unfitted_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        dmkde.Mahalanobis().score_samples(rng.normal(size=(3, 8)))
    with pytest.raises(RuntimeError):
        _core.Mahalanobis().score_batch(np.zeros((3, 8)))


@pytest.mark.parametrize("method", ["score_samples", "decision_function", "predict"])
def test_F12_1d_input_rejected(method):
    m = dmkde.DMKDE(feature_dim=32).fit(rng.normal(size=(100, 1)))
    m.calibrate(rng.normal(size=(100, 1)))
    with pytest.raises(ValueError, match="2D"):
        getattr(m, method)(rng.normal(size=100))


# ---------------------------------------------------------------------------
# 2. Non-finite input  (audit F3, F7; review2 p02, p08, p19)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_F3_p02_nonfinite_training_row_rejected(bad):
    Xb = X.copy()
    Xb[7, 2] = bad
    with pytest.raises(ValueError):
        dmkde.DMKDE(feature_dim=64).fit(Xb)
    with pytest.raises(ValueError):
        _core.DMKDE(5, 64, 1.0, 0).fit(Xb)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_p02_nonfinite_partial_fit_rejected_and_model_intact(bad):
    m = fitted()
    before = m.score_samples(X[:5])
    row = X[:1].copy()
    row[0, 1] = bad
    with pytest.raises(ValueError):
        m.partial_fit(row, alpha=0.01)
    with pytest.raises(ValueError):
        m._impl.update(np.ascontiguousarray(row[0]), 0.01)
    np.testing.assert_array_equal(m.score_samples(X[:5]), before)


def test_p02_core_update_nan_alpha_rejected():
    c = _core.DMKDE(5, 16, 1.0, 0)
    c.fit(X)
    with pytest.raises(ValueError):
        c.update(np.zeros(5), float("nan"))
    assert abs(c.trace - 1.0) < 1e-12


def test_p02_nan_query_raises():
    m = fitted()
    q = X[:3].copy()
    q[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m.score_samples(q)
    with pytest.raises(ValueError):
        m._impl.score_batch(q)


def test_p02_mahalanobis_nonfinite_fit_rejected():
    Xb = X.copy()
    Xb[3, 1] = np.nan
    with pytest.raises(ValueError):
        dmkde.Mahalanobis().fit(Xb)
    with pytest.raises(ValueError):
        _core.Mahalanobis().fit(Xb)


def test_p02_mahalanobis_nan_query_never_most_normal():
    mh = dmkde.Mahalanobis().fit(X)
    T = np.vstack([X[:5], [[np.nan, 0, 0, 0, 0]]])
    with pytest.raises(ValueError):
        mh.score_samples(T)
    with pytest.raises(ValueError):
        mh._impl.score_batch(T)


def test_F7_p19_roc_auc_nan_rejected():
    with pytest.raises(ValueError):
        dmkde.roc_auc(np.array([0.9, np.nan, 0.8]), np.array([0.1, 0.65]))
    with pytest.raises(ValueError):
        dmkde.roc_auc(np.array([0.9, 0.8]), np.array([np.inf, 0.65]))
    with pytest.raises(ValueError):
        _core.roc_auc(np.array([0.9, np.nan]), np.array([0.1]))


def test_p08_roc_auc_empty_rejected():
    with pytest.raises(ValueError):
        dmkde.roc_auc([], [1.0, 2.0])
    with pytest.raises(ValueError):
        _core.roc_auc(np.array([]), np.array([1.0]))


def test_p08_core_roc_auc_strided_view():
    from sklearn.metrics import roc_auc_score

    base = np.arange(10, dtype=float)
    view = base[::2]
    expected = roc_auc_score(np.r_[np.ones(5), np.zeros(1)], np.r_[view, 5.0])
    assert _core.roc_auc(view, np.array([5.0])) == pytest.approx(expected)
    assert dmkde.roc_auc(view, [5.0]) == pytest.approx(expected)


def test_p08_roc_auc_2d_rejected():
    with pytest.raises(ValueError):
        dmkde.roc_auc(np.array([[1.0, 2.0], [3.0, 4.0]]), [0.0])
    with pytest.raises(ValueError):
        _core.roc_auc(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.0]))


# ---------------------------------------------------------------------------
# 3. Parameters  (audit F2, F4; review2 p04, p09)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma", [0.0, -1.5, float("nan"), float("inf")])
def test_F2_p04_invalid_sigma_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        dmkde.DMKDE(feature_dim=64, sigma=sigma).fit(X)
    with pytest.raises(ValueError, match="sigma"):
        _core.DMKDE(5, 64, sigma, 0)


def test_p04_core_zero_input_dim_rejected():
    with pytest.raises(ValueError, match="input_dim"):
        _core.DMKDE(0, 64, 1.0, 0)


def test_p04_feature_dim_cap():
    run_isolated("""
try:
    _core.DMKDE(3, (1 << 15) + 2, 1.0, 0)
except ValueError as e:
    assert "allow_large" in str(e), e
else:
    raise AssertionError("no ValueError")
try:
    dmkde.DMKDE(feature_dim=(1 << 15) + 2).fit(np.zeros((2, 3)))
except ValueError as e:
    assert "allow_large_feature_dim" in str(e), e
else:
    raise AssertionError("no ValueError")
print("OK")""", limit_memory=True)


def test_p04_core_update_before_fit_raises():
    c = _core.DMKDE(3, 64, 1.0, 0)
    with pytest.raises(RuntimeError, match="before fit"):
        c.update(np.zeros(3), 0.1)
    with pytest.raises(RuntimeError, match="before fit"):
        c.score_batch(np.zeros((1, 3)))


def test_p04_random_state_accepts_none_int_randomstate_generator():
    for rs in (None, 7, np.random.RandomState(0), np.random.default_rng(0)):
        m = dmkde.DMKDE(feature_dim=16, random_state=rs).fit(X)
        assert m.score_samples(X[:2]).shape == (2,)
    a = dmkde.DMKDE(feature_dim=16, random_state=np.random.RandomState(3)).fit(X)
    b = dmkde.DMKDE(feature_dim=16, random_state=np.random.RandomState(3)).fit(X)
    np.testing.assert_array_equal(a.score_samples(X[:5]), b.score_samples(X[:5]))
    with pytest.raises(ValueError):
        dmkde.DMKDE(feature_dim=16, random_state=-1).fit(X)


def test_p04_calibrate_contamination_validated():
    m = fitted()
    for c in (1.5, 0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            m.calibrate(X, contamination=c)


def test_F4_p09_mahalanobis_needs_two_rows():
    with pytest.raises(ValueError, match="n_samples=1"):
        dmkde.Mahalanobis().fit(X[:1])
    with pytest.raises(ValueError):
        _core.Mahalanobis().fit(X[:1])


def test_p09_mahalanobis_ridge_is_scale_invariant():
    r = np.random.default_rng(0)
    L = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0], [0, 0, 0.5, 0.2]])
    tr = r.normal(size=(2000, 4)) @ L.T
    tn = r.normal(size=(500, 4)) @ L.T
    ta = tn.copy()
    ta[:, 1] += 0.5 * r.choice([-1, 1], 500)
    aucs, scores = [], []
    for c in (1.0, 1e-2, 1e-4):
        m = dmkde.Mahalanobis().fit(tr * c)
        aucs.append(dmkde.roc_auc(m.score_samples(tn * c), m.score_samples(ta * c)))
        scores.append(m.score_samples(tn[:20] * c))
    assert max(aucs) - min(aucs) < 1e-9, aucs
    for s in scores[1:]:
        np.testing.assert_allclose(s, scores[0], rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. sklearn contract  (audit F5, F10; review2 p05)
# ---------------------------------------------------------------------------
def test_F5_p05_get_params_and_clone():
    from sklearn.base import clone

    est = dmkde.DMKDE(feature_dim=64, sigma=2.0, contamination=0.1)
    assert est.get_params()["sigma"] == 2.0
    c = clone(est)
    assert c.get_params() == est.get_params()
    est.set_params(sigma=3.0)
    assert est.sigma == 3.0


def test_F5_p05_pickle_roundtrip_identical():
    m = fitted()
    m.partial_fit(X[:10], alpha=0.05)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.score_samples(X[:20]), m.score_samples(X[:20]))
    np.testing.assert_array_equal(m2._impl.rho_matrix(), m._impl.rho_matrix())
    assert m2.offset_ == m.offset_
    assert m2._impl.n_streamed == 10 and m2._impl.n_fitted == len(X)


def test_F5_pickle_uses_saved_weights_not_seed():
    m = fitted()
    state = m.__getstate__()
    np.testing.assert_array_equal(state["_core_state"]["W"], m._impl.weights())
    state["_core_state"]["seed"] = 12345           # seed is informational only
    m2 = dmkde.DMKDE.__new__(dmkde.DMKDE)
    m2.__setstate__(state)
    np.testing.assert_array_equal(m2.score_samples(X[:10]), m.score_samples(X[:10]))


def test_F5_core_state_import_is_validated():
    c = _core.DMKDE(5, 16, 1.0, 0)
    c.fit(X)
    good = c.get_state()
    for key, bad in [("W", np.zeros((3, 5))), ("rho", np.full((16, 16), np.nan)),
                     ("n_fitted", -1), ("feature_dim", 15)]:
        st = dict(good)
        st[key] = bad
        with pytest.raises(ValueError):
            _core.DMKDE.from_state(st)
    asym = good["rho"].copy()
    asym[0, 1] += 0.1
    with pytest.raises(ValueError):
        _core.DMKDE.from_state(dict(good, rho=asym))


def test_p05_pipeline_fit():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = make_pipeline(StandardScaler(), dmkde.DMKDE(feature_dim=32)).fit(X)
    assert p.predict(X).shape == (len(X),)


def test_p05_decision_function_sign_convention():
    # The RFF weights come from std::normal_distribution, whose sequence differs between
    # standard libraries (MSVC vs libstdc++/libc++), so a fixed seed gives a different model
    # per platform. Use a large D and a fixed local RNG so the check does not hinge on one draw.
    local = np.random.default_rng(1234)
    Xtr = local.normal(size=(400, 5))
    m = dmkde.DMKDE(feature_dim=512, sigma=1.0).fit(Xtr)
    Z = np.vstack([local.normal(size=(50, 5)), local.normal(size=(50, 5)) + 5])
    df, pr, s = m.decision_function(Z), m.predict(Z), m.score_samples(Z)
    np.testing.assert_allclose(df, s - m.offset_)
    np.testing.assert_array_equal(pr, np.where(df >= 0, 1, -1))
    assert (df[50:] < 0).mean() > 0.9


def test_F10_predict_works_after_fit_and_docstring_is_accurate():
    # offset_ is the k-th smallest leave-one-out training score,
    # k = floor((N + 1)·c) (rev5b M1), so exactly k - 1 of them fall below it.
    m = fitted()
    loo = dmkde._loo_training_scores(m.score_samples(X), len(X))
    k = int(np.floor((len(X) + 1) * m.contamination))
    assert np.sum(loo < m.offset_) == k - 1
    assert abs(np.mean(loo < m.offset_) - m.contamination) <= 1.0 / len(X) + 1e-12
    assert "5th-percentile of training-set scores" not in (dmkde.DMKDE.predict.__doc__ or "")


def test_p05_fit_resets_calibrated_threshold():
    m = fitted()
    m.calibrate(X, contamination=0.4)
    X2 = rng.normal(size=(300, 5)) * 0.1
    m.fit(X2)
    loo = dmkde._loo_training_scores(m.score_samples(X2), len(X2))
    k = int(np.floor((len(X2) + 1) * m.contamination))
    assert m.offset_ == np.sort(loo)[k - 1]
    fresh = rng.normal(size=(5000, 5)) * 0.1
    assert abs(np.mean(m.predict(fresh) == -1) - 0.05) < 0.02


def test_p05_unfitted_raises_notfittederror():
    from sklearn.exceptions import NotFittedError

    for meth in ("score_samples", "decision_function", "predict"):
        with pytest.raises(NotFittedError):
            getattr(dmkde.DMKDE(), meth)(X)


def test_p05_partial_fit_works_as_first_call():
    m = dmkde.DMKDE(feature_dim=32).partial_fit(X)
    ref = dmkde.DMKDE(feature_dim=32).fit(X)
    np.testing.assert_array_equal(m.score_samples(X[:5]), ref.score_samples(X[:5]))


def test_p05_core_fit_resets_n_streamed():
    c = _core.DMKDE(5, 16, 1.0, 0)
    c.fit(X)
    c.update(np.zeros(5), 0.1)
    c.fit(X)
    assert c.n_streamed == 0


# DMKDE is a novelty detector: offset_ is an order statistic of the
# leave-one-out training scores (rev5 t07, rev5b M1), so predict() on the
# training set itself flags fewer than `contamination` of it. These two checks
# require the opposite; scikit-learn skips the first for
# LocalOutlierFactor(novelty=True) and does not offer fit_predict there for
# the second. Overriding fit_predict to threshold the leave-one-out scores
# (rev4b #4) would not help: the check first asserts fit_predict(X) ==
# fit(X).predict(X).
DMKDE_EXPECTED_FAILED_CHECKS = {
    "check_outliers_train":
        "offset_ is calibrated on leave-one-out scores for new data, so predict "
        "on the training set flags fewer than contamination of it",
    "check_outliers_fit_predict":
        "fit_predict == fit().predict() (the check itself requires that), which "
        "labels the training set against a threshold calibrated for new data",
}


def test_p05_check_estimator():
    """sklearn's full estimator check suite passes; DMKDE has exactly the two
    documented novelty-detector exceptions, and they really fail (so the list
    cannot hide anything else)."""
    import warnings

    from sklearn.utils.estimator_checks import check_estimator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_estimator(dmkde.Mahalanobis())
        res = check_estimator(dmkde.DMKDE(feature_dim=32), on_fail=None)
    failed = {r["check_name"] for r in res if r["status"] == "failed"}
    assert failed == set(DMKDE_EXPECTED_FAILED_CHECKS), failed
    assert len(res) > 40


# ---------------------------------------------------------------------------
# 5. PyOD adapter  (review2 p06)
# ---------------------------------------------------------------------------
def test_p06_pyod_class_cached_isinstance_and_pickle():
    dp = pytest.importorskip("dmkde.pyod")
    pytest.importorskip("pyod")
    assert dp.DMKDEDetector is dp.DMKDEDetector
    det = dp.DMKDEDetector(feature_dim=64, contamination=0.05).fit(X)
    assert isinstance(det, dp.DMKDEDetector)
    det2 = pickle.loads(pickle.dumps(det))
    np.testing.assert_array_equal(det2.decision_function(X[:10]), det.decision_function(X[:10]))


def test_p06_pyod_predict_proba():
    pytest.importorskip("pyod")
    from dmkde.pyod import DMKDEDetector

    det = DMKDEDetector(feature_dim=64, contamination=0.05).fit(X)
    p = det.predict_proba(rng.normal(size=(4, 5)) + 5)
    assert p.shape == (4, 2)
    np.testing.assert_allclose(p.sum(1), 1.0)


def test_p06_pyod_unfitted_decision_function_raises():
    pytest.importorskip("pyod")
    from sklearn.exceptions import NotFittedError

    from dmkde.pyod import DMKDEDetector

    with pytest.raises(NotFittedError):
        DMKDEDetector().decision_function(X)


def test_p06_pyod_width_checked():
    pytest.importorskip("pyod")
    run_isolated(EXPECT_VALUE_ERROR + """
from dmkde.pyod import DMKDEDetector
det = DMKDEDetector(feature_dim=64).fit(X)
expect(lambda: det.decision_function(X[:3, :2]), "features")
print("OK")""")


# ---------------------------------------------------------------------------
# 6. Qiskit backend  (review2 p07)
# ---------------------------------------------------------------------------
def test_p07_qiskit_rereads_rho_after_partial_fit():
    pytest.importorskip("qiskit")
    from dmkde.qiskit_backend import QiskitDMKDE

    m = dmkde.DMKDE(feature_dim=16, sigma=1.5).fit(X[:, :4])
    q = QiskitDMKDE(m)
    m.partial_fit(rng.normal(size=(50, 4)) + 3, alpha=0.05)
    T = rng.normal(size=(5, 4))
    for t in T:
        assert abs(q.score(t) - m.score_samples(t[None])[0]) < 1e-10
    m.fit(rng.normal(size=(80, 4)) * 2)
    for t in T:
        assert abs(q.score(t) - m.score_samples(t[None])[0]) < 1e-10


def test_p07_qiskit_encode_width_checked():
    pytest.importorskip("qiskit")
    run_isolated(EXPECT_VALUE_ERROR + """
from dmkde.qiskit_backend import QiskitDMKDE
m = dmkde.DMKDE(feature_dim=16, sigma=1.5).fit(X[:, :4])
q = QiskitDMKDE(m)
expect(lambda: q.encode(np.zeros(3)), "features")
print("OK")""")


def test_p07_qiskit_docs_make_no_false_claims():
    pytest.importorskip("qiskit")
    import dmkde.qiskit_backend as qb

    doc = qb.__doc__
    assert "via `StatevectorEstimator`" not in doc
    assert "swap in a real backend" not in doc
    assert "no hardware path" in doc
    readme = read("README.md")
    assert "statevector estimator" not in readme


# ---------------------------------------------------------------------------
# 7. GIL  (review2 p18)
# ---------------------------------------------------------------------------
def test_p18_gil_released_during_score_batch():
    m = dmkde.DMKDE(feature_dim=1024, sigma=1.0).fit(rng.normal(size=(50, 4)))
    Q = rng.normal(size=(1500, 4))
    done = threading.Event()
    started = threading.Event()

    def work():
        started.set()
        m.score_samples(Q)
        done.set()

    t0 = time.perf_counter()
    m.score_samples(Q[:300])
    single = (time.perf_counter() - t0) * 5     # rough duration of the full call
    th = threading.Thread(target=work)
    th.start()
    started.wait()
    ticks = 0
    t_start = time.perf_counter()
    while not done.is_set():
        ticks += 1
        if ticks % 1000 == 0:
            time.sleep(0)
    elapsed = time.perf_counter() - t_start
    th.join()
    # With the GIL held for the whole C++ call the main thread could not run
    # at all while it was in progress; with it released it spins freely.
    assert elapsed > 0.2 * single
    assert ticks / max(elapsed, 1e-9) > 1e4, (ticks, elapsed)


# ---------------------------------------------------------------------------
# 8. Docs and metadata  (audit F6; review2 p11, p16, p18, p20)
# ---------------------------------------------------------------------------
AUTHORS_2201 = ["Useche", "Bustos-Brinez", "Gallego-Mejia", "Gonz"]


def test_p20_citations_list_all_arxiv_authors():
    readme, cff, hdr = read("README.md"), read("CITATION.cff"), read("include/dmkde.hpp")
    for text in (readme, cff, hdr, dmkde.__doc__):
        for a in ("Useche", "Bustos-Brinez", "Gallego-Mejia"):
            assert a in text, a
    # InQMAD (2210.05061) first author is Gallego-Mejia, not Useche.
    assert not re.search(r"useche\S*inqmad", readme, re.I)
    assert re.search(r"Gallego-Mejia[^}]*\n?[^}]*InQMAD|InQMAD[\s\S]{0,300}Gallego-Mejia", readme)
    # AD-DMKDE (2210.14796) first author is Bustos-Brinez.
    m = re.search(r"@misc\{(\w+),[^@]*AD-DMKDE", readme)
    assert m and "bustos" in m.group(1).lower()


def test_p20_inqmad_year_is_2022():
    hdr = read("include/dmkde.hpp")
    assert "arXiv:2210.05061 (2022)" in hdr
    assert "2210.05061 (2025)" not in hdr


def test_F6_p11_docs_state_effective_bandwidth_and_unnormalized():
    for text in (read("README.md"), read("include/dmkde.hpp"),
                 dmkde.DMKDE.__doc__, dmkde.__doc__):
        assert "√2" in text or "sqrt(2)" in text
        assert "unnormalized" in text.lower()


def test_p18_p20_no_false_performance_claims():
    for rel in ("README.md", "include/dmkde.hpp", "python/bindings.cpp", "BENCHMARKS.md"):
        text = read(rel)
        assert "parallelizable with OpenMP" not in text, rel
        assert "zero-copy" not in text, rel
        assert "O(1)-per-sample" not in text, rel
        assert "Streaming O(1) updates" not in text, rel


def test_p16_readme_pypi_checkbox():
    assert "- [ ] PyPI release" not in read("README.md")


def test_p16_release_workflow_hardened():
    wf = read(".github/workflows/wheels.yml")
    uses = re.findall(r"uses:\s*(\S+)", wf)
    assert uses and all(re.search(r"@[0-9a-f]{40}$", u) for u in uses), uses
    assert re.search(r"^permissions:", wf, re.M)
    assert "id-token: write" in wf
    assert "password:" not in wf and "PYPI_API_TOKEN" not in wf
    assert re.search(r"environment:\s*\n\s*name:\s*pypi", wf)
    assert "pyproject" in wf and "GITHUB_REF_NAME" in wf      # tag == version check
    assert "pytest" in read("pyproject.toml") + wf
    manifest = read("MANIFEST.in")
    assert re.search(r"^recursive-include tests ", manifest, re.M)


def test_version_bumped_to_0_1_1():
    assert dmkde.__version__ == "0.1.1"


# ---------------------------------------------------------------------------
# 9. Third review  (rev3 p1, p2, p3, p4/p10, p6, p8, p12)
# ---------------------------------------------------------------------------
HUGE = 1.7e308          # finite (np.isfinite is True) but overflows the RFF phase


def test_rev3_p1_huge_finite_query_raises_instead_of_nan():
    m = fitted()
    Q = X[:5].copy()
    Q[0, 0] = HUGE
    assert np.isfinite(Q).all()
    for fn in (m.score_samples, m.decision_function, m.predict):
        with pytest.raises(ValueError, match="not finite"):
            fn(Q)
    with pytest.raises(ValueError, match="not finite"):
        m._impl.score(Q[0])
    with pytest.raises(ValueError, match="not finite"):
        m._impl.score_batch(Q)
    # calibrate on data containing such a row raises and keeps offset_.
    off = m.offset_
    with pytest.raises(ValueError, match="not finite"):
        m.calibrate(np.vstack([X, Q]))
    assert m.offset_ == off and np.isfinite(off)
    assert (m.predict(X) == -1).sum() < len(X) // 2


def test_rev3_p1_mahalanobis_huge_query_never_nan():
    M = dmkde.Mahalanobis().fit(X[:, :3])
    s = M.score_samples(np.array([[1e308, -1e308, 0.0]]))
    assert np.isfinite(s).all() and s[0] < -1e300, s        # most anomalous, not NaN
    # A query whose difference from the mean overflows raises.
    for q in ([[HUGE, -HUGE, HUGE]], [[-HUGE, 0.0, 0.0]]):
        try:
            s = M.score_samples(np.array(q))
        except ValueError as e:
            assert "not finite" in str(e)
        else:
            assert np.isfinite(s).all(), s
    # Normal queries are unchanged.
    T = rng.normal(size=(20, 3))
    assert np.isfinite(M.score_samples(T)).all()


def test_rev3_p2_partial_fit_is_atomic():
    m = dmkde.DMKDE(feature_dim=32).fit(X[:, :2])
    rho0 = m._impl.rho_matrix().copy()
    n0, rev0 = m._impl.n_streamed, m._impl.revision
    B = rng.normal(size=(10, 2))
    B[-1, 0] = HUGE
    with pytest.raises(ValueError, match="not finite"):
        m.partial_fit(B, alpha=0.5)
    assert m._impl.n_streamed == n0 and m._impl.revision == rev0
    np.testing.assert_array_equal(m._impl.rho_matrix(), rho0)
    # A valid batch gives exactly the same rho as row-by-row core updates.
    G = rng.normal(size=(10, 2))
    ref = _core.DMKDE.from_state(m._impl.get_state())
    for row in G:
        ref.update(np.ascontiguousarray(row), 0.5)
    m.partial_fit(G, alpha=0.5)
    np.testing.assert_array_equal(m._impl.rho_matrix(), ref.rho_matrix())
    assert m._impl.n_streamed == n0 + 10


def test_rev3_p3_failed_refit_keeps_previous_fit():
    m = dmkde.DMKDE(feature_dim=16).fit(X[:, :3])
    before = (m.n_features_in_, m.offset_, m._impl)
    T = rng.normal(size=(4, 3))
    s0 = m.score_samples(T)
    Y = rng.normal(size=(50, 5))
    Y[0, 0] = HUGE
    with pytest.raises(ValueError):
        m.fit(Y)
    assert (m.n_features_in_, m.offset_, m._impl) == before
    np.testing.assert_array_equal(m.score_samples(T), s0)
    # First fit failing leaves the estimator unfitted.
    u = dmkde.DMKDE(feature_dim=16)
    with pytest.raises(ValueError):
        u.fit(Y)
    assert not hasattr(u, "n_features_in_") and not hasattr(u, "_impl")
    # Mahalanobis: refit that the core rejects (zero variance) keeps the old fit.
    M = dmkde.Mahalanobis().fit(X[:, :3])
    sm = M.score_samples(T)
    with pytest.raises(ValueError):
        M.fit(np.ones((10, 4)))
    assert M.n_features_in_ == 3
    np.testing.assert_array_equal(M.score_samples(T), sm)


def test_rev3_p4_dmkde_state_rejects_non_psd_rho():
    c = _core.DMKDE(2, 8, 1.0, 0)
    c.fit(X[:, :2])
    st = c.get_state()
    bad = np.zeros((8, 8))
    bad[0, 0], bad[1, 1] = 5.0, -4.0                 # symmetric, trace 1, not PSD
    with pytest.raises(ValueError, match="positive semidefinite"):
        _core.DMKDE.from_state(dict(st, rho=bad))
    # Through pickle as well.
    m = dmkde.DMKDE(feature_dim=8).fit(X[:, :2])
    state = m.__getstate__()
    state["_core_state"]["rho"] = bad
    with pytest.raises(ValueError, match="positive semidefinite"):
        dmkde.DMKDE.__new__(dmkde.DMKDE).__setstate__(state)
    W = st["W"].copy()
    W[0, 0] = np.inf
    with pytest.raises(ValueError):
        _core.DMKDE.from_state(dict(st, W=W))


def test_rev3_p4_valid_states_still_round_trip():
    # Rank-deficient (N < D), streamed and large-D models all re-import.
    for n, D, k in [(3, 64, 0), (200, 32, 40), (500, 512, 5)]:
        c = _core.DMKDE(5, D, 0.8, 3)
        c.fit(X[:n])
        for row in rng.normal(size=(k, 5)):
            c.update(row, 0.05)
        r = _core.DMKDE.from_state(c.get_state())
        np.testing.assert_array_equal(r.score_batch(X[:20]), c.score_batch(X[:20]))
    # Mahalanobis on near-collinear features re-imports.
    a = rng.normal(size=(300, 1))
    Z = np.hstack([a, a + 1e-6 * rng.normal(size=(300, 1)), rng.normal(size=(300, 1))])
    M = dmkde.Mahalanobis().fit(Z)
    M2 = pickle.loads(pickle.dumps(M))
    np.testing.assert_array_equal(M2.score_samples(Z[:20]), M.score_samples(Z[:20]))


@pytest.mark.parametrize("inv_cov", [
    -np.eye(2),                                       # negative definite
    np.array([[1.0, 0.0], [0.0, 0.0]]),               # singular
    np.array([[1.0, 2.0], [2.0, 1.0]]),               # indefinite
    np.array([[1.0, 0.5], [-0.5, 1.0]]),              # not symmetric
    np.array([[1.0, np.nan], [np.nan, 1.0]]),         # non-finite
])
def test_rev3_p4_mahalanobis_state_requires_spd(inv_cov):
    M = dmkde.Mahalanobis().fit(X[:, :2])
    st = M._impl.get_state()
    with pytest.raises(ValueError):
        _core.Mahalanobis.from_state(dict(st, inv_cov=inv_cov))
    with pytest.raises(ValueError):
        _core.Mahalanobis.from_state(dict(st, mean=np.array([np.inf, 0.0])))


@pytest.mark.parametrize("key,value", [
    ("seed", -1), ("seed", 2**64), ("seed", "1"), ("format", 1.0), ("format", 2),
    ("input_dim", 2**40), ("input_dim", 1.5), ("feature_dim", -8), ("sigma", "1.0"),
    ("sigma", None), ("fitted", "yes"), ("n_fitted", 2**70), ("n_streamed", [1]),
    ("allow_large", "no"), ("W", "abc"), ("rho", None),
])
def test_rev3_p4_malformed_dmkde_state_raises_valueerror(key, value):
    c = _core.DMKDE(2, 8, 1.0, 0)
    c.fit(X[:, :2])
    st = c.get_state()
    st[key] = value
    with pytest.raises(ValueError):
        _core.DMKDE.from_state(st)


@pytest.mark.parametrize("state", [
    None, [], {"format": 1}, {"format": "1", "mean": np.zeros(2), "inv_cov": np.eye(2)},
    {"format": 1, "mean": "x", "inv_cov": np.eye(2)},
])
def test_rev3_p4_malformed_states_raise_valueerror(state):
    with pytest.raises(ValueError):
        _core.Mahalanobis.from_state(state)
    if isinstance(state, dict) and "mean" not in state:
        with pytest.raises(ValueError):
            _core.DMKDE.from_state(state)
    if state is None or isinstance(state, list):
        with pytest.raises(ValueError):
            _core.DMKDE.from_state(state)


def test_rev3_p12_update_not_starved_by_concurrent_scoring():
    m = _core.DMKDE(4, 512, 1.0, 1)
    m.fit(rng.normal(size=(50, 4)))
    Q = rng.normal(size=(200, 4))
    t = time.perf_counter()
    m.score_batch(Q)
    batch = time.perf_counter() - t
    window = max(4.0, 40 * batch)               # readers run this long at most
    bound = max(1.0, 8 * batch)                 # a writer waits ~1 batch per reader
    stop = threading.Event()
    deadline = time.time() + window

    def reader():
        while not stop.is_set() and time.time() < deadline:
            m.score_batch(Q)

    rs = [threading.Thread(target=reader) for _ in range(3)]
    for r in rs:
        r.start()
    try:
        time.sleep(min(0.3, window / 10))
        lat = []
        for _ in range(5):
            t = time.perf_counter()
            m.update(Q[0], 0.01)
            lat.append(time.perf_counter() - t)
    finally:
        stop.set()
        for r in rs:
            r.join()
    assert max(lat) < bound, (lat, bound, batch)


def test_rev3_p8_qiskit_uses_fitted_feature_dim():
    pytest.importorskip("qiskit")
    from dmkde.qiskit_backend import QiskitDMKDE

    m = dmkde.DMKDE(feature_dim=12).fit(X[:, :2])     # fitted D = 12
    m.set_params(feature_dim=16)                       # parameter only
    with pytest.raises(ValueError, match="12"):
        QiskitDMKDE(m)
    m2 = dmkde.DMKDE(feature_dim=16).fit(X[:, :2])
    m2.set_params(feature_dim=12)
    q = QiskitDMKDE(m2)
    assert q.n_qubits == 4
    assert abs(q.score(X[0, :2]) - m2.score_samples(X[:1, :2])[0]) < 1e-10
    q.to_circuit(X[0, :2])
    # A refit to a different D after wrapping is reported, not mis-simulated.
    m2.set_params(feature_dim=8).fit(X[:, :2])
    with pytest.raises(ValueError, match="feature_dim"):
        q.score(X[0, :2])


def test_rev3_p6_changelog_claims_are_accurate():
    log = read("CHANGELOG.md")
    assert "results\n  are unchanged;" not in log and "are unchanged; the published" not in log
    assert "1e-7 relative" in log and "2e-2" in log and "near-collinear" in log
    assert "PYPI_API_TOKEN" in log and "deleted" in log
    assert "required reviewer" in log and "`v*`" in log


def test_rev3_release_checks_all_version_strings():
    wf = read(".github/workflows/wheels.yml")
    assert "python/dmkde/__init__.py" in wf and "CITATION.cff" in wf
    cff = read("CITATION.cff")
    assert re.search(r'^version:\s*"?%s"?\s*$' % re.escape(dmkde.__version__), cff, re.M)


# ---------------------------------------------------------------------------
# 10. Fourth and fifth reviews  (rev4: code/fuzz, rev5: user/docs)
# ---------------------------------------------------------------------------
ASAN = "asan" in os.environ.get("LD_PRELOAD", "")


def _quiet_fit(est, Xf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(Xf)


def test_rev5_t07_offset_is_exact_leave_one_out():
    """offset_ is an order statistic of (N·s - 1)/(N - 1), which equals the
    score of each training row under the model refitted without it."""
    r = np.random.default_rng(70)
    Xs = r.normal(size=(25, 3))
    n, D, sg, seed = len(Xs), 64, 1.3, 5
    full = _core.DMKDE(3, D, sg, seed)
    full.fit(Xs)
    s = full.score_batch(Xs)
    explicit = np.empty(n)
    for i in range(n):
        m = _core.DMKDE(3, D, sg, seed)            # same seed => same RFF weights
        m.fit(np.delete(Xs, i, axis=0))
        explicit[i] = m.score_batch(Xs[i:i + 1])[0]
    loo = dmkde._loo_training_scores(s, n)
    np.testing.assert_allclose(loo, explicit, rtol=0, atol=1e-13)
    est = _quiet_fit(dmkde.DMKDE(feature_dim=D, sigma=sg, random_state=seed,
                                 contamination=0.1), Xs)
    k = int(np.floor((n + 1) * 0.1))                  # rev5b M1: k-th smallest
    assert est.offset_ == pytest.approx(float(np.sort(explicit)[k - 1]), abs=1e-13)
    # score_samples is untouched: it is still the in-sample Born-rule score.
    np.testing.assert_array_equal(est.score_samples(Xs), s)


T07_CONFIGS = [  # (d, N, sigma, D) from review round 5, t07
    (4, 1000, 1.0, 256), (4, 200, 1.0, 256), (4, 1000, 0.5, 256),
    (8, 1000, 1.0, 256), (8, 1000, 1.5, 256), (10, 2000, 1.0, 1024),
    (20, 1000, 1.0, 256), (20, 1000, 3.0, 256),
]


@pytest.mark.parametrize("d,n,sg,D", T07_CONFIGS)
def test_rev5_t07_fresh_false_alarm_rate_matches_contamination(d, n, sg, D):
    r = np.random.default_rng(600 + d + n + D + int(10 * sg))
    Xf, F = r.normal(size=(n, d)), r.normal(size=(5000, d))
    m = _quiet_fit(dmkde.DMKDE(feature_dim=D, sigma=sg, contamination=0.05), Xf)
    rate = float((m.predict(F) == -1).mean())
    # 0.1.1-rc (in-sample quantile): 0.06-1.00 on these configurations.
    assert 0.025 <= rate <= 0.10, rate


def test_rev5_t07_pyod_threshold_uses_leave_one_out_scores():
    pytest.importorskip("pyod")
    from dmkde.pyod import DMKDEDetector
    r = np.random.default_rng(71)
    Xf, F = r.normal(size=(1000, 8)), r.normal(size=(5000, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = DMKDEDetector(feature_dim=256, sigma=1.0, contamination=0.05).fit(Xf)
    rate = float(det.predict(F).mean())
    assert 0.025 <= rate <= 0.10, rate               # rc: ~0.49
    assert det.labels_.mean() == pytest.approx(0.05, abs=0.002)
    s = det._model._impl.score_batch(Xf)
    np.testing.assert_allclose(det.decision_scores_, -(len(Xf) * s - 1) / (len(Xf) - 1))


def test_rev5_t07_single_sample_fit_has_zero_offset_and_warns():
    with pytest.warns(UserWarning, match="single sample"):
        m = dmkde.DMKDE(feature_dim=16).fit(X[:1])
    assert m.offset_ == 0.0
    assert np.all(m.predict(X[:20]) == 1)
    m.calibrate(X[1:50])
    assert m.offset_ > 0.0


@pytest.mark.parametrize("d,warn", [(2, False), (4, False), (20, True), (100, True)])
def test_rev5_t06_noise_floor_warning(d, warn):
    r = np.random.default_rng(60 + d)
    Xf = r.normal(size=(1000, d))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dmkde.DMKDE().fit(Xf)                          # default sigma = 1
    hits = [x for x in w if "noise floor" in str(x.message)]
    assert bool(hits) == warn, [str(x.message) for x in w]
    if warn:
        assert issubclass(hits[0].category, UserWarning)
        # Following the advice (sigma ~ sqrt(d)) silences it.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dmkde.DMKDE(sigma=np.sqrt(d)).fit(Xf)


def test_rev5_t06_docs_explain_sigma_scaling_and_noise_floor():
    for text in (dmkde.DMKDE.__doc__, read("README.md")):
        assert "standardize" in text.lower()
        assert "sqrt(d)" in text or "√d" in text
        assert "noise floor" in text
        assert "1/feature_dim" in text or "1/D" in text
    assert re.search(r"default .*sigma.* 0\.2\.0|0\.2\.0.*default.*sigma", read("CHANGELOG.md"),
                     re.IGNORECASE | re.DOTALL)


MAHA_FACTORS = [1e-4, 1e-2, 1.0, 1e2, 1e4]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rev5_t10_mahalanobis_per_feature_scale_invariant(seed):
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(100 + seed)
    L = np.array([[1, .5, 0, 0, 0], [0, 1, .3, 0, 0], [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, .2], [0, 0, 0, 0, 1.]])
    Xm, N = r.normal(size=(2000, 5)) @ L, r.normal(size=(500, 5)) @ L
    A = r.normal(size=(500, 5)) @ L
    A[:, 0] += 3
    y = np.r_[np.ones(500), np.zeros(500)]
    ref = dmkde.Mahalanobis().fit(Xm)
    s_ref = np.r_[ref.score_samples(N), ref.score_samples(A)]
    auc_ref = roc_auc_score(y, s_ref)
    for _ in range(4):
        sc = r.choice(MAHA_FACTORS, size=5)
        m = dmkde.Mahalanobis().fit(Xm * sc)
        s = np.r_[m.score_samples(N * sc), m.score_samples(A * sc)]
        np.testing.assert_allclose(s, s_ref, rtol=1e-9)
        assert roc_auc_score(y, s) == auc_ref
        rank = np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(s_ref)))[0, 1]
        assert rank == pytest.approx(1.0, abs=1e-12)


def test_rev5_t10_mahalanobis_constant_feature_still_handled():
    Xc = np.c_[X[:, :2], np.full(len(X), 3.0)]
    m = dmkde.Mahalanobis().fit(Xc)
    on = m.score_samples(np.array([[0.0, 0.0, 3.0]]))[0]
    off = m.score_samples(np.array([[0.0, 0.0, 3.5]]))[0]
    assert np.isfinite(on) and np.isfinite(off) and off < on - 10
    with pytest.raises(ValueError, match="zero"):
        dmkde.Mahalanobis().fit(np.ones((10, 3)))


def _psd_state(D, eigs, seed=0):
    """A fitted-looking DMKDE state whose rho has the given spectrum."""
    c = _core.DMKDE(2, D, 1.0, 0)
    c.fit(X[:5, :2])
    st = c.get_state()
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(D, D)))
    rho = (q * np.asarray(eigs)) @ q.T
    st["rho"] = (rho + rho.T) / 2
    return st


@pytest.mark.parametrize("D", [2, 8, 62, 64, 66, 130, 258])
def test_rev4_blocked_cholesky_matches_spectrum(D):
    """The blocked PSD check accepts/rejects exactly by the smallest eigenvalue
    (tolerance 1e-9·trace) across panel boundaries."""
    for lam_min, ok in [(0.0, True), (1e-12, True), (-5e-10, True),
                        (-3e-9, False), (-1e-4, False), (-0.2, False)]:
        eigs = np.full(D, 1.0 / D)
        eigs[-1] = lam_min
        eigs[0] += 1.0 - eigs.sum()                    # trace 1
        st = _psd_state(D, eigs, seed=D)
        if ok:
            _core.DMKDE.from_state(st)
        else:
            with pytest.raises(ValueError, match="positive semidefinite"):
                _core.DMKDE.from_state(st)


def _corrupted_large_state(D):
    """A real fitted state, minus 0.01·yyᵀ plus 0.01·xxᵀ: trace 1, symmetric,
    every diagonal entry and 2x2 minor fine, smallest eigenvalue ≈ -0.01."""
    c = _core.DMKDE(4, D, 1.0, 1)
    c.fit(np.random.default_rng(3).normal(size=(200, 4)))
    st = c.get_state()
    r = np.random.default_rng(4)
    x, y = r.normal(size=D), r.normal(size=D)
    x /= np.linalg.norm(x)
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    st["rho"] = st["rho"] + 0.01 * (np.outer(x, x) - np.outer(y, y))
    return c, st


@pytest.mark.parametrize("D", [2048, 2050])               # Cholesky path / probe path
def test_rev4_large_state_psd_check_both_paths(D):
    c, bad = _corrupted_large_state(D)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _core.DMKDE.from_state(bad)
    # Valid states, including rank-1 after alpha=1 and streamed ones, load.
    good = _core.DMKDE.from_state(c.get_state())
    np.testing.assert_array_equal(good.score_batch(X[:5, :4]), c.score_batch(X[:5, :4]))
    c.update(np.ones(4), 1.0)
    _core.DMKDE.from_state(c.get_state())
    assert _core.MAX_FEATURE_DIM >= 4096


@pytest.mark.skipif(ASAN, reason="timing is meaningless under ASan")
def test_rev4_from_state_load_time():
    """D = 4096 used to take ~19-37 s to load (unblocked O(D³) Cholesky)."""
    c = _core.DMKDE(4, 4096, 1.0, 0)
    c.fit(np.random.default_rng(0).normal(size=(200, 4)))
    st = c.get_state()
    t = time.perf_counter()
    _core.DMKDE.from_state(st)
    dt = time.perf_counter() - t
    assert dt < 6.0, dt
    small = _core.DMKDE(4, 1024, 1.0, 0)
    small.fit(np.random.default_rng(0).normal(size=(200, 4)))
    st = small.get_state()
    t = time.perf_counter()
    _core.DMKDE.from_state(st)
    assert time.perf_counter() - t < 0.5


def _spin_rate(seconds: float = 0.1) -> float:
    ticks, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ticks += 1
    return ticks / seconds


def test_rev4_mahalanobis_from_state_releases_gil():
    d = 1600
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(d, d)))
    P = (q * np.linspace(1.0, 2.0, d)) @ q.T
    st = {"format": 1, "mean": np.zeros(d), "inv_cov": (P + P.T) / 2}
    _core.Mahalanobis.from_state(st)
    rate = _spin_rate()
    done = threading.Event()

    def work():
        for _ in range(3):
            _core.Mahalanobis.from_state(st)
        done.set()

    th = threading.Thread(target=work)
    ticks, t0 = 0, time.perf_counter()
    th.start()
    while not done.is_set():
        ticks += 1
        time.perf_counter()
    elapsed = time.perf_counter() - t0
    th.join()
    # Holding the GIL through the O(d³) check, the main thread only ran in
    # the gaps between calls; with it released it spins throughout.
    assert ticks / elapsed > 0.3 * rate, (ticks, elapsed, rate)


def test_rev4_readers_not_starved_by_busy_writers():
    """With 3 threads calling update() back to back, score() still gets in
    after about one write (0.1.1-rc: a single score() waited ~8-10 s)."""
    m = _core.DMKDE(4, 1024, 1.0, 1)
    m.fit(rng.normal(size=(20, 4)))
    x = np.zeros(4)
    t = time.perf_counter()
    m.update(x, 0.01)
    one = time.perf_counter() - t
    stop = time.time() + 4.0

    def writer():
        while time.time() < stop:
            m.update(x, 0.01)

    ws = [threading.Thread(target=writer) for _ in range(3)]
    for w in ws:
        w.start()
    waits = []
    try:
        time.sleep(0.2)
        while time.time() < stop - 0.5:
            t = time.perf_counter()
            m.score(x)
            waits.append(time.perf_counter() - t)
    finally:
        for w in ws:
            w.join()
    assert len(waits) >= 20, (len(waits), one)
    assert max(waits) < max(1.0, 40 * one), (max(waits), one)


def test_rev4_fallback_tracks_feature_names():
    pytest.importorskip("pandas")
    code = textwrap.dedent("""
        import sys, warnings
        sys.modules["sklearn"] = None              # block scikit-learn
        import numpy as np, pandas as pd, dmkde
        assert not dmkde._HAVE_SKLEARN
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50) * 100})
        m = dmkde.DMKDE(feature_dim=16, sigma=5.0).fit(df)
        assert list(m.feature_names_in_) == ["a", "b"]
        for bad in (df[["b", "a"]], df[["a"]], df.rename(columns={"b": "c"})):
            try:
                m.score_samples(bad)
            except ValueError as e:
                assert "feature names should match" in str(e)
            else:
                raise AssertionError("accepted %r" % list(bad.columns))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.score_samples(df.values)
        assert any("fitted with feature names" in str(x.message) for x in w)
        m.fit(df.values)
        assert not hasattr(m, "feature_names_in_")
        print("OK")
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, timeout=300, check=False)
    assert p.returncode == 0 and "OK" in p.stdout, p.stdout + p.stderr


def test_rev4_rev5_docs():
    readme, bench, log = read("README.md"), read("BENCHMARKS.md"), read("CHANGELOG.md")
    hdr = read("include/dmkde.hpp")
    for text in (readme, bench):
        assert "wins outright" not in text
        assert "1:1" not in text and "hardware offload" not in text
    assert re.search(r"single.split", bench, re.IGNORECASE) and "seed" in bench
    assert "OCSVM" in bench or "OneClassSVM" in bench
    assert re.search(r"seeds? 1.{0,6}2", bench) and re.search(r"seeds? 0.{0,6}9", bench)
    assert "7 of these 10" in bench and "categorical" in bench
    assert re.search(r"statevector simulation", bench, re.IGNORECASE)
    assert re.search(r"no hyperparameter search", bench) is None or "sensitiv" in bench
    assert "0.836" not in readme and "0.439 " not in readme and "200 EMA" not in readme
    assert "1/alpha" in dmkde.DMKDE.partial_fit.__doc__ or "1/α" in dmkde.DMKDE.partial_fit.__doc__
    assert "-inf" in hdr
    assert "leave-one-out" in log and "phase-fair" in log


# ---------------------------------------------------------------------------
# 11. Review rounds 4b (code/fuzz) and 5b (user/docs) on 0.1.1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("other_std", [1.0, 1e3, 1e5, 1e7])
def test_rev4b_1_mahalanobis_constant_feature_floor_is_absolute(other_std):
    """A feature constant in training that changes at test time is far out,
    whatever the units of the other columns (rc: AUC 0.65 at std 1e3, 0.50
    at 1e5 and 1e7, because the floor was 1e-6 x their mean variance)."""
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(3)
    n = 2000
    Xtr = np.column_stack([r.normal(size=(n, 3)) * other_std, np.zeros(n)])
    m = dmkde.Mahalanobis().fit(Xtr)
    norm = np.column_stack([r.normal(size=(500, 3)) * other_std, np.zeros(500)])
    flag = norm.copy()
    flag[:, 3] = 1.0
    auc = roc_auc_score(np.r_[np.zeros(500), np.ones(500)],
                        -np.r_[m.score_samples(norm), m.score_samples(flag)])
    assert auc == 1.0
    # P_ii = 1e6 for the constant feature: a deviation of 1 alone is distance 1000.
    q = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert m.score_samples(q)[0] == pytest.approx(-1000.0, rel=1e-6)


def test_rev4b_1_constant_feature_score_independent_of_other_units():
    r = np.random.default_rng(4)
    base = np.column_stack([r.normal(size=(2000, 3)), np.zeros(2000)])
    q = np.array([[0.0, 0, 0, 1.0]])
    # rc: -987, -17.1, -0.176, -0.0404 for column-0 units x1, 1e2, 1e4, 1e6.
    s = [dmkde.Mahalanobis().fit(base * [a, 1, 1, 1]).score_samples(q)[0]
         for a in (1, 1e2, 1e4, 1e6)]
    np.testing.assert_allclose(s, s[0], rtol=1e-9)
    assert s[0] == pytest.approx(-1000.0, rel=1e-3)
    # Non-constant features stay unit-invariant with a constant column present.
    Y = np.column_stack([r.normal(size=(50, 3)) * 2, np.zeros(50)])
    ref = dmkde.Mahalanobis().fit(base).score_samples(Y)
    for a in ([1e-4, 1, 1e4, 1], [3.0, 1e-8, 1e8, 1]):
        np.testing.assert_allclose(
            dmkde.Mahalanobis().fit(base * a).score_samples(Y * a), ref, rtol=1e-9)
    assert "absolute" in dmkde.Mahalanobis.__doc__


@pytest.mark.parametrize("tiny", [1e-155, 1e-160, 1e-170, 1e-300])
def test_rev4b_6_mahalanobis_tiny_scale_feature_rejected_as_underflow(tiny):
    """rc: 1e-155/1e-160 were rejected as 'overflowed (training data too
    large)' and 1e-170/1e-300 were silently treated as constant (scores off
    by 84%)."""
    r = np.random.default_rng(1)
    Z = r.normal(size=(200, 4))
    Z[:, 0] *= tiny
    with pytest.raises(ValueError, match="underflow.*feature 0"):
        dmkde.Mahalanobis().fit(Z)


def test_rev4b_6_mahalanobis_extreme_but_representable_scales_ok():
    r = np.random.default_rng(1)
    Xm, Y = r.normal(size=(200, 4)), r.normal(size=(50, 4)) * 2
    base = dmkde.Mahalanobis().fit(Xm).score_samples(Y)
    for a in ([1e-150, 1, 1e150, 3], [1e-154, 1, 1, 1], [1e-100] * 4):
        a = np.array(a)
        s = dmkde.Mahalanobis().fit(Xm * a).score_samples(Y * a)
        np.testing.assert_allclose(s, base, rtol=1e-12)
    with pytest.raises(ValueError, match="overflow.*too large"):
        dmkde.Mahalanobis().fit(Xm * 1e300)


def _mean_fresh_rate(N, reps, d=2, sg=1.0, c=0.05, D=128, nf=1000, detector="dmkde"):
    rates = []
    for r in range(reps):
        g = np.random.default_rng(10_000 * N + r)
        Xf, F = g.normal(size=(N, d)), g.normal(size=(nf, d))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if detector == "pyod":
                from dmkde.pyod import DMKDEDetector
                det = DMKDEDetector(feature_dim=D, sigma=sg, contamination=c,
                                    random_state=r).fit(Xf)
                rates.append(np.mean(det.predict(F) == 1))
            else:
                m = dmkde.DMKDE(feature_dim=D, sigma=sg, contamination=c,
                                random_state=r).fit(Xf)
                rates.append(np.mean(m.predict(F) == -1))
    rates = np.asarray(rates)
    return rates.mean(), rates.std() / np.sqrt(reps)


@pytest.mark.parametrize("N,reps", [(2, 100), (5, 100), (20, 300), (200, 60)])
def test_rev5b_m1_fresh_false_alarm_rate_at_most_contamination(N, reps):
    """Mean fresh flag rate <= c (+ Monte-Carlo error) for every N.
    ef05caf (np.quantile): 0.37 (N=2), 0.20 (N=5), 0.085 (N=20)."""
    mean, se = _mean_fresh_rate(N, reps)
    assert mean <= 0.05 + 3 * se + 1e-12, (mean, se)


def test_rev5b_m1_fresh_false_alarm_rate_about_c_at_large_n():
    mean, se = _mean_fresh_rate(1000, 30)
    assert abs(mean - 0.05) <= max(3 * se, 0.004), (mean, se)


def test_rev5b_m1_threshold_is_order_statistic_and_warns():
    r = np.random.default_rng(5)
    # k = floor((N + 1)·c) = 0 for N = 10 at c = 0.05: nothing can be flagged.
    with pytest.warns(UserWarning, match="too few to set a threshold"):
        m = dmkde.DMKDE(feature_dim=64, sigma=2.0).fit(r.normal(size=(10, 2)))
    assert m.offset_ == 0.0
    assert np.all(m.predict(r.normal(size=(500, 2)) * 5) == 1)
    # N·c < 5 but k >= 1: a threshold, with a warning that the rate is noisy.
    Xs = r.normal(size=(40, 2))
    with pytest.warns(UserWarning, match="noisy"):
        m = dmkde.DMKDE(feature_dim=64, sigma=2.0).fit(Xs)
    loo = dmkde._loo_training_scores(m._impl.score_batch(Xs), 40)
    assert m.offset_ == np.sort(loo)[int(np.floor(41 * 0.05)) - 1]
    # calibrate uses the same rule on the held-out scores.
    H = r.normal(size=(300, 2))
    m.calibrate(H, contamination=0.1)
    assert m.offset_ == np.sort(m.score_samples(H))[int(np.floor(301 * 0.1)) - 1]
    with pytest.warns(UserWarning, match="too few"):
        m.calibrate(H[:5])
    assert m.offset_ == 0.0
    # No warning at all for a comfortable N.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dmkde.DMKDE(feature_dim=64, sigma=2.0).fit(r.normal(size=(200, 2)))
    for doc in (dmkde.DMKDE.__doc__, read("README.md"), read("CHANGELOG.md")):
        assert "floor((N + 1)" in doc or "floor((n + 1)" in doc
        assert "fresh normal data" in doc


def test_rev5b_m1_pyod_threshold_uses_same_order_statistic():
    pytest.importorskip("pyod")
    from dmkde.pyod import DMKDEDetector
    r = np.random.default_rng(72)
    Xf = r.normal(size=(100, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = DMKDEDetector(feature_dim=64, sigma=2.0, contamination=0.05).fit(Xf)
    assert det.threshold_ == -det._model.offset_
    assert det.labels_.sum() == int(np.floor(101 * 0.05)) - 1
    mean, se = _mean_fresh_rate(20, 300, detector="pyod")
    assert mean <= 0.05 + 3 * se, (mean, se)         # ef05caf: ~0.09


@pytest.mark.parametrize("trial,d,D,n,row", [
    (23, 2, 512, 89, [-1.4227417685154136, 0.25845279091298756]),
    (80, 2, 512, 46, [5.334375570673462, 110.0734095024795]),
])
def test_rev4b_3_duplicate_training_rows_not_all_flagged(trial, d, D, n, row):
    """ef05caf: LOO rounded an ulp above s, so offset_ > every score and
    predict flagged 100% of the training rows."""
    Xd = np.tile(np.array(row), (n, 1))
    m = _quiet_fit(dmkde.DMKDE(feature_dim=D, sigma=1.0, random_state=trial), Xd)
    s = m.score_samples(Xd)
    assert m.offset_ <= s.min()
    assert np.all(m.predict(Xd) == 1)
    ss = np.full(4, 1.0000000000000004)
    assert np.all(dmkde._loo_training_scores(ss, 4) <= ss)


PROBE_DRAWS_CPP = r"""
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    size_t n = std::strtoul(argv[1], nullptr, 10);
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull);  // ef05caf's fixed probe seed
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<double> v(n);
    FILE* f = std::fopen(argv[2], "wb");
    for (int t = 0; t < 5; ++t) { for (auto& e : v) e = g(rng); std::fwrite(v.data(), 8, n, f); }
    std::fclose(f);
}
"""


@pytest.mark.skipif(ASAN, reason="slow under ASan")
def test_rev4b_2_psd_probe_not_bypassable_with_known_seed(tmp_path):
    """ef05caf seeded the D > 2048 probe with a fixed public constant, so a
    negative eigenvector chosen orthogonal to its 4 probes and Lanczos start
    was accepted deterministically. The probe is now seeded per call."""
    import shutil
    cxx = shutil.which("g++") or shutil.which("c++")
    if cxx is None or sys.platform == "win32":
        pytest.skip("needs a C++ compiler with the same <random> as the extension")
    src, exe, out = tmp_path / "draws.cpp", tmp_path / "draws", tmp_path / "draws.bin"
    src.write_text(PROBE_DRAWS_CPP)
    subprocess.run([cxx, "-O1", "-std=c++17", "-o", str(exe), str(src)], check=True,
                   timeout=120)
    D = 2050
    subprocess.run([str(exe), str(D), str(out)], check=True, timeout=60)
    draws = np.fromfile(out).reshape(5, D)
    r = np.random.default_rng(0)
    Xtr = r.normal(size=(50, 3))
    c = _core.DMKDE(3, D, 2.0, 42, True)
    c.fit(Xtr)
    st = c.get_state()
    A = np.array(st["rho"])
    Phi = np.array([c.transform(x) for x in Xtr])
    B, _ = np.linalg.qr(np.vstack([Phi, draws]).T)
    u = c.transform(np.array([4.0, -4.0, 4.0]))
    u = u - B @ (B.T @ u)
    u /= np.linalg.norm(u)
    delta = 1e-4                         # 1e5 x the tolerance; 2x2 minors pass
    rho = (A - delta * np.outer(u, u)) / (1 - delta)
    st["rho"] = (rho + rho.T) / 2
    for _ in range(3):
        with pytest.raises(ValueError, match="positive semidefinite"):
            _core.DMKDE.from_state(st)
    hdr = read("include/dmkde.hpp")
    assert "0x9E3779B97F4A7C15ull)) {" not in hdr and "fresh_probe_seed()" in hdr
    assert "NOT adversarially robust" in hdr and "adversarial" in read("CHANGELOG.md")


def test_rev5b_noise_floor_warning_not_misleading_at_tiny_feature_dim():
    """At D = 6 (Qiskit) the rc message claimed 'the kernel between typical
    points is ~0' for a median score 2.2x the floor. It now states the ratio
    and names feature_dim as a possible cause."""
    r = np.random.default_rng(12)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dmkde.DMKDE(feature_dim=6, sigma=1.5).fit(r.normal(size=(200, 4)))
    msgs = [str(x.message) for x in w if "noise floor" in str(x.message)]
    assert msgs, [str(x.message) for x in w]
    assert "is ~0" not in msgs[0]
    assert "x 1/feature_dim" in msgs[0] and "feature_dim is too small" in msgs[0]
    assert "after standardizing" in msgs[0]


def test_rev5b_sparse_input_raises_typeerror_as_documented():
    sp = pytest.importorskip("scipy.sparse")
    m = fitted()
    with pytest.raises(TypeError, match="[Ss]parse"):
        m.score_samples(sp.csr_matrix(X[:3]))
    with pytest.raises(TypeError, match="[Ss]parse"):
        dmkde.DMKDE().fit(sp.csr_matrix(X))
    readme = read("README.md")
    assert "Sparse" in readme and "TypeError" in readme


def test_rev5b_docs():
    readme, bench, log = read("README.md"), read("BENCHMARKS.md"), read("CHANGELOG.md")
    # Noise-floor sweep stated with its N and the measured losses.
    assert "N = 500" in log and "0.07-0.18" in log and "N = 10" in log and "0.09" in log
    assert "1.2-2.0 lost 0.02-0.05" not in log
    # Credit-card PR-AUC: DMKDE 4th of 5, 0.10 behind LOF; ROC above LOF / OCSVM.
    assert "4th of 5" in bench and "0.10" in bench
    assert "close to OneClassSVM" not in bench
    # The outdated competition write-up is no longer shipped (round 5c).
    assert not (ROOT / "SUBMISSION.md").exists() and "SUBMISSION.md" not in readme
    # sigma rule says "standardized".
    for text in (readme, dmkde.DMKDE.__doc__):
        assert re.search(r"median\s+pairwise\s+distance\s+of\s+the\s+\*?standardized", text)
    # Earlier density-matrix RFF work credited.
    for text in (readme, read("CITATION.cff")):
        assert "2102.04394" in text and "Toledo-Cort" in text and "Vargas-Calder" in text
    # Mahalanobis decision_function documented.
    assert "no ``offset_``" in dmkde.Mahalanobis.__doc__
    assert "decision_function" in readme and "offset" in readme
    # check_estimator claim exact.
    assert "`DMKDE` and `Mahalanobis` pass `sklearn.utils.estimator_checks.check_estimator`." \
        not in log
    assert "absolute" in log


# --- Round 4c: Mahalanobis constant columns whose value is not exactly summable ---
@pytest.mark.parametrize("value", [0.1, 3.3, 1732492800.123, float(np.float32(0.1)), 1e300, -7e-300])
@pytest.mark.parametrize("n", [50, 1000, 20000])
def test_rev4c_1_constant_column_any_value(value, n):
    """A constant column must add nothing to training-row distances and give a
    deviation delta about 1e6*delta^2, whatever the constant and n (0.1.1 rc
    summed the mean naively: at n >= 1000 a constant 0.1 got P_ii ~ 5e29)."""
    rng = np.random.default_rng(n)
    X = np.c_[rng.normal(size=n), np.full(n, value)]
    m = dmkde.Mahalanobis().fit(X)
    base = dmkde.Mahalanobis().fit(X[:, :1])
    np.testing.assert_allclose(m.score_samples(X[:100]), base.score_samples(X[:100, :1]),
                               rtol=1e-9, atol=1e-12)
    if abs(value) < 1e10:
        x = np.array([[0.0, value]]); y = np.array([[0.0, value + 1e-3]])
        # score = -distance; 1e6 * (1e-3)^2 = 1 extra squared distance, up to the
        # rounding of value + 1e-3
        diff = m.score_samples(y)[0] ** 2 - m.score_samples(x)[0] ** 2
        assert 0.99 < diff < 1.01, diff


def test_rev4c_lo1_tiny_contamination_warns_not_overflows():
    rng = np.random.default_rng(0)
    for c in (1e-310, 5e-324):
        with pytest.warns(UserWarning, match="more than 1e308"):
            m = dmkde.DMKDE(contamination=c).fit(rng.normal(size=(30, 2)))
        assert m.offset_ == 0.0


# --- Round 4d: columns constant only up to rounding at large magnitude ---
@pytest.mark.parametrize("value", [1.7e12, 1.7e15, 1.7e18, 1e300])
@pytest.mark.parametrize("ulps", [1, 3, 40])
def test_rev4d_1_rounding_constant_large_magnitude(value, ulps):
    """A column whose spread is a few ulp of a huge value must not swamp the
    real features (7a0f46a gave it inverse variance 1e6: training distance
    ~2.6e5 at 1.7e18 with 3 ulp)."""
    rng = np.random.default_rng(ulps)
    n = 2000
    col = value + rng.integers(0, ulps + 1, n) * np.spacing(value)
    assert col.min() != col.max()
    X = np.c_[rng.normal(size=(n, 3)), col]
    A = np.c_[rng.normal(size=(200, 3)) + 4, np.full(200, value)]
    if col.std() > 1e154:
        # A real spread whose std is beyond the documented ~1e154 limit is
        # refused loudly rather than mis-scaled.
        with pytest.raises(ValueError, match="overflowed: feature 3"):
            dmkde.Mahalanobis().fit(X)
        return
    m = dmkde.Mahalanobis().fit(X)
    s_train = m.score_samples(X[:200])
    assert np.median(-s_train) < 10, np.median(-s_train)
    s = m.score_samples(np.r_[X[200:400], A])
    y = np.r_[np.zeros(200), np.ones(200)]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, -s) > 0.99


# --- Round 5d: tiny real spread at large magnitude is variance, not "constant" ---
@pytest.mark.parametrize("make_col, anom_in_col", [
    (lambda r, n: 1e14 + r.normal(size=n), False),
    (lambda r, n: np.round(1.7e18 + r.normal(0, 3000, n)), False),
    (lambda r, n: 1.7e9 + r.normal(0, 1e-5, n), True),
    (lambda r, n: 1 + 1e-14 * r.normal(size=n), True),
    (lambda r, n: 1e15 + r.normal(size=n), True),
])
def test_rev5d_1_small_relative_spread_uses_real_variance(make_col, anom_in_col):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7)
    n = 2000
    col = make_col(rng, n)
    X = np.c_[rng.normal(size=n), col]
    T = X[rng.choice(n, 300)].copy()
    A = X[rng.choice(n, 300)].copy()
    if anom_in_col:
        A[:, 1] += 5 * col.std()
    else:
        A[:, 0] += 5
    s = dmkde.Mahalanobis().fit(X).score_samples(np.r_[T, A])
    assert roc_auc_score(np.r_[np.zeros(300), np.ones(300)], -s) > 0.99


def test_rev5d_1b_large_offset_unit_noise_fits():
    """1e15 + N(0,1) in every column was rejected as 'zero variance in every feature'."""
    rng = np.random.default_rng(0)
    X = 1e15 + rng.normal(size=(1000, 3))
    m = dmkde.Mahalanobis().fit(X)
    assert np.all(np.isfinite(m.score_samples(X[:10])))
