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
    m = fitted()
    frac = np.mean(m.predict(X) == -1)
    assert abs(frac - m.contamination) <= 1.0 / len(X) + 1e-12
    assert "5th-percentile of training-set scores" not in (dmkde.DMKDE.predict.__doc__ or "")


def test_p05_fit_resets_calibrated_threshold():
    m = fitted()
    m.calibrate(X, contamination=0.4)
    X2 = rng.normal(size=(300, 5)) * 0.1
    m.fit(X2)
    assert m.offset_ == pytest.approx(np.quantile(m.score_samples(X2), m.contamination))
    assert abs(np.mean(m.predict(X2) == -1) - 0.05) < 0.01


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


def test_p05_check_estimator():
    """sklearn's full estimator check suite passes with no expected failures."""
    import warnings

    from sklearn.utils.estimator_checks import check_estimator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for est in (dmkde.DMKDE(feature_dim=32), dmkde.Mahalanobis()):
            check_estimator(est)


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

