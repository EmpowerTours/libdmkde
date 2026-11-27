"""Regression tests for the 0.2.0 LOW-severity fixes (items B4-B23 of the 0.2.0 plan).

Each test reproduces the 0.1.1 behaviour it guards against: it failed on the 0.1.1
code and passes on 0.2.0. Documentation items are checked by asserting that the
documented text exists and that the behaviour it describes is real.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

import dmkde
from dmkde import _core

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(2026)
X3 = rng.normal(size=(200, 3))


def fitted(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dmkde.DMKDE(feature_dim=32, sigma=1.7, **kw).fit(X3)


# ---------------------------------------------------------------- B4 partial_fit alpha
@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, np.nan, np.inf, "0.5", True])
def test_B4_partial_fit_on_unfitted_estimator_validates_alpha(alpha):
    est = dmkde.DMKDE(feature_dim=16)
    with pytest.raises(ValueError, match="alpha"):
        est.partial_fit(X3, alpha=alpha)
    assert not hasattr(est, "_impl") and not hasattr(est, "n_features_in_")


@pytest.mark.parametrize("alpha", ["0.5", True, 1 + 0j])
def test_B4_partial_fit_alpha_must_be_a_real_number(alpha):
    est = fitted()
    with pytest.raises(ValueError, match="alpha"):
        est.partial_fit(X3[:5], alpha=alpha)


# ---------------------------------------------------------------- B5 noise-floor warning
def _noise_data():
    return np.random.default_rng(5).normal(size=(200, 20))


def test_B5_noise_floor_warning_as_error_leaves_estimator_unfitted():
    est = dmkde.DMKDE(feature_dim=16, sigma=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="noise floor"):
            est.fit(_noise_data())
    for attr in ("_impl", "offset_", "n_features_in_"):
        assert not hasattr(est, attr), attr


def test_B5_noise_floor_warning_as_error_keeps_previous_fit():
    est = fitted()
    before = est.score_samples(X3[:10]).copy()
    offset = est.offset_
    est.set_params(sigma=0.1, feature_dim=16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="noise floor"):
            est.fit(_noise_data())
    assert est.n_features_in_ == 3 and est.offset_ == offset
    assert np.array_equal(est.score_samples(X3[:10]), before)


def test_B5_noise_floor_warning_points_at_the_caller():
    with pytest.warns(UserWarning, match="noise floor") as rec:
        dmkde.DMKDE(feature_dim=16, sigma=0.1).fit(_noise_data())
    assert any(Path(w.filename).resolve() == Path(__file__).resolve()
               for w in rec if "noise floor" in str(w.message))


# ---------------------------------------------------------------- B6 transform
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_B6_core_transform_rejects_non_finite_input(bad):
    m = _core.DMKDE(2, 8, 1.0, 1)
    with pytest.raises(ValueError, match="NaN or inf"):
        m.transform(np.array([bad, 0.0]))


def test_B6_core_transform_rejects_overflowing_embedding():
    m = _core.DMKDE(2, 8, 1.0, 1)
    with pytest.raises(ValueError, match="not finite"):
        m.transform(np.array([1.7e308, 1.7e308]))
    assert np.all(np.isfinite(m.transform(np.array([0.3, -0.2]))))


# ---------------------------------------------------------------- B7 predict threshold
@pytest.mark.parametrize("thr", [np.nan, np.inf, -np.inf, "0.1", True, 1 + 0j])
def test_B7_predict_rejects_non_finite_or_non_real_threshold(thr):
    with pytest.raises(ValueError, match="threshold"):
        fitted().predict(X3[:5], threshold=thr)


def test_B7_predict_with_finite_threshold_still_works():
    est = fitted()
    s = est.score_samples(X3[:20])
    t = float(np.median(s))
    assert np.array_equal(est.predict(X3[:20], threshold=t), np.where(s >= t, 1, -1))
    assert np.array_equal(est.predict(X3[:20], threshold=np.float32(t)),
                          np.where(s - float(np.float32(t)) >= 0, 1, -1))


# ---------------------------------------------------------------- B8 pickle validation
def _state():
    return fitted().__getstate__()


@pytest.mark.parametrize("offset", [np.nan, np.inf, -0.5, 2.0, "0.1", None, True])
def test_B8_unpickling_rejects_bad_offset(offset):
    st = _state()
    st["offset_"] = offset
    with pytest.raises(ValueError, match="offset_"):
        dmkde.DMKDE.__new__(dmkde.DMKDE).__setstate__(st)


def test_B8_unpickling_rejects_missing_offset():
    st = _state()
    del st["offset_"]
    with pytest.raises(ValueError, match="offset_"):
        dmkde.DMKDE.__new__(dmkde.DMKDE).__setstate__(st)


@pytest.mark.parametrize("nf", [4, 0, -3, 3.0, "3", None])
def test_B8_unpickling_rejects_inconsistent_n_features_in(nf):
    st = _state()
    st["n_features_in_"] = nf
    with pytest.raises(ValueError, match="n_features_in_"):
        dmkde.DMKDE.__new__(dmkde.DMKDE).__setstate__(st)


@pytest.mark.parametrize("nf", [4, 2.0, None])
def test_B8_mahalanobis_unpickling_rejects_inconsistent_n_features_in(nf):
    st = dmkde.Mahalanobis().fit(X3).__getstate__()
    st["n_features_in_"] = nf
    with pytest.raises(ValueError, match="n_features_in_"):
        dmkde.Mahalanobis.__new__(dmkde.Mahalanobis).__setstate__(st)


def test_B8_valid_pickles_still_round_trip():
    est = fitted()
    est2 = pickle.loads(pickle.dumps(est))
    assert est2.offset_ == est.offset_
    assert np.array_equal(est2.score_samples(X3), est.score_samples(X3))
    m = dmkde.Mahalanobis().fit(X3)
    assert np.array_equal(pickle.loads(pickle.dumps(m)).score_samples(X3), m.score_samples(X3))
    u = pickle.loads(pickle.dumps(dmkde.DMKDE()))  # unfitted: no offset_ needed
    assert not hasattr(u, "_impl")


# ---------------------------------------------------------------- B9 calibrate too few
def test_B9_calibrate_on_too_few_rows_raises_and_keeps_offset():
    est = fitted()
    offset = est.offset_
    with pytest.raises(ValueError, match="too few"):
        est.calibrate(X3[:5])  # k = floor(6 * 0.05) = 0
    assert est.offset_ == offset
    assert offset > 0.0


# ---------------------------------------------------------------- B10 saturated scores
def test_B10_saturated_scores_warn():
    X = np.random.default_rng(3).normal(size=(300, 4))
    with pytest.warns(UserWarning, match="saturated"):
        dmkde.DMKDE(feature_dim=64, sigma=1000.0).fit(X)


def test_B10_scores_rounding_to_one_warn_that_detection_is_lost():
    X = np.random.default_rng(3).normal(size=(300, 4))
    with pytest.warns(UserWarning, match="rounding"):
        dmkde.DMKDE(feature_dim=64, sigma=1e9).fit(X)


def test_B10_no_saturation_warning_for_a_sensible_sigma():
    X = np.random.default_rng(3).normal(size=(300, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dmkde.DMKDE(feature_dim=256, sigma=2.0).fit(X)


# ---------------------------------------------------------------- B11 PyOD contamination
def test_B11_pyod_default_contamination_is_documented_as_deliberate():
    src = (ROOT / "python" / "dmkde" / "pyod.py").read_text()
    assert "contamination=0.1" in src
    assert "PyOD's own default" in src and "DMKDE.contamination" in src


# ---------------------------------------------------------------- B12 score range
def test_B12_scores_never_exceed_one():
    hi = []
    for seed in range(40):
        x = np.random.default_rng(seed).normal(size=(1, 3))
        for D in (2, 4, 8, 16, 64):
            m = _core.DMKDE(3, D, 1.0, seed)
            m.fit(np.repeat(x, 3, axis=0))
            hi.append(m.score_batch(x)[0])
            hi.append(m.score(x[0]))
    hi = np.array(hi)
    assert np.all(hi <= 1.0) and np.all(hi >= 0.0)
    assert np.any(hi == 1.0)


# ---------------------------------------------------------------- B13 correlated tiny
def test_B13_correlated_tiny_features_error_names_the_cause():
    r = np.random.default_rng(0)
    z = r.normal(size=500)
    X = np.c_[z, z + 1e-3 * r.normal(size=500)] * 1e-152
    with pytest.raises(ValueError, match=r"features 0 and 1.*correlat"):
        dmkde.Mahalanobis().fit(X)


# ---------------------------------------------------------------- B15 DBL_MAX distance
def test_B15_distance_beyond_dbl_max_raises_and_is_documented():
    m = dmkde.Mahalanobis().fit(X3)
    with pytest.raises(ValueError, match="not finite"):
        m.score_samples(np.array([[1.7e308, -1.7e308, 1.7e308]]))
    assert "DBL_MAX" in dmkde.Mahalanobis.__doc__
    assert "DBL_MAX" in (ROOT / "README.md").read_text()


# ---------------------------------------------------------------- B16 positional alpha
@pytest.mark.parametrize("y", [np.array(0.5), np.float64(0.5), 0.5, np.array([0.5]), [0.5],
                               [0.1, 0.2]])
def test_B16_alpha_passed_positionally_is_rejected(y):
    est = fitted()
    rho = est._impl.rho_matrix()
    with pytest.raises(TypeError, match="alpha by keyword"):
        est.partial_fit(X3[:5], y)
    assert np.array_equal(est._impl.rho_matrix(), rho)


def test_B16_alpha_positional_rejected_on_unfitted_estimator_too():
    est = dmkde.DMKDE(feature_dim=16)
    with pytest.raises(TypeError, match="alpha by keyword"):
        est.partial_fit(X3[:5], np.array(0.5))
    assert not hasattr(est, "_impl")


def test_B16_a_real_y_is_still_accepted_and_ignored():
    est = fitted()
    est.partial_fit(X3[:5], np.zeros(5), alpha=0.1)
    est.partial_fit(X3[:5], list(range(5)), alpha=0.1)
    assert est._impl.n_streamed == 10


# ---------------------------------------------------------------- B17 complex
C = np.array([[1 + 5j, 0.0, 0.0]])


@pytest.mark.parametrize("call", [
    lambda: dmkde.roc_auc(np.array([1 + 1j, 2.0]), np.array([0.5])),
    lambda: _core.roc_auc(np.array([1 + 1j, 2.0]), np.array([0.5])),
    lambda: fitted()._impl.score_batch(C),
    lambda: fitted()._impl.score(C[0]),
    lambda: fitted()._impl.transform(C[0]),
    lambda: fitted()._impl.update(C[0], 0.1),
    lambda: fitted()._impl.update_batch(C, 0.1),
    lambda: _core.DMKDE(3, 8, 1.0, 1).fit(C),
    lambda: _core.Mahalanobis().fit(np.vstack([C, C + 1, C - 2])),
    lambda: dmkde.Mahalanobis().fit(X3)._impl.score_batch(C),
    lambda: dmkde.Mahalanobis().fit(X3)._impl.score(C[0]),
    lambda: fitted().score_samples(C),
])
def test_B17_complex_input_rejected(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        with pytest.raises(ValueError, match="[Cc]omplex"):
            call()


# ---------------------------------------------------------------- B18 huge Python ints
BIG = 10**400


@pytest.mark.parametrize("call", [
    lambda: dmkde.DMKDE().fit([[BIG, 1.0], [1.0, 2.0]]),
    lambda: dmkde.Mahalanobis().fit([[BIG, 1.0], [1.0, 2.0], [3.0, 1.0]]),
    lambda: dmkde.DMKDE(feature_dim=16).fit(X3[:, :2]).score_samples([[BIG, 1.0]]),
    lambda: dmkde.roc_auc([BIG], [1.0]),
    lambda: _core.roc_auc([BIG], [1.0]),
    lambda: _core.DMKDE(2, 8, 1.0, 1).fit([[BIG, 1.0]]),
    lambda: _core.DMKDE(2, 8, 1.0, 1).fit(np.array([[BIG, 1.0]], dtype=object)),
    lambda: _core.Mahalanobis().fit([[BIG, 1.0], [1.0, 2.0]]),
])
def test_B18_huge_python_int_raises_valueerror(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError):
            call()


def test_B18_core_non_numeric_input_is_valueerror_not_cast_error():
    with pytest.raises(ValueError):
        _core.DMKDE(2, 8, 1.0, 1).fit("abc")
    with pytest.raises(ValueError):
        _core.DMKDE(2, 8, 1.0, 1).fit([["a", "b"]])


# ---------------------------------------------------------------- B19 input_dim cap
def test_B19_core_input_dim_capped_unless_allow_large():
    with pytest.raises(ValueError, match="allow_large"):
        _core.DMKDE(2**22, 256)  # W would be 128 x 4M doubles = 4 GiB
    with pytest.raises(ValueError, match="allow_large"):
        _core.DMKDE(2**31 - 1, 2)
    assert _core.MAX_WEIGHT_COUNT == 2**28


def test_B19_state_claiming_huge_input_dim_rejected():
    st = fitted()._impl.get_state()
    st["input_dim"] = 2**22
    st["feature_dim"] = 256
    with pytest.raises(ValueError, match="allow_large"):
        _core.DMKDE.from_state(st)


# ---------------------------------------------------------------- B20 tolerances
def test_B20_from_state_tolerances_documented():
    doc = _core.DMKDE.from_state.__doc__
    readme = (ROOT / "README.md").read_text()
    for text in (doc, readme):
        assert "1e-6" in text and "1e-12" in text and "1e-9" in text


# ---------------------------------------------------------------- B21 duplicate rows
def test_B21_duplicate_rows_over_flag_and_calibrate_fixes_it():
    r = np.random.default_rng(21)
    base = r.normal(size=(150, 6))
    Xdup = np.repeat(base, 3, axis=0)  # every row three times
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = dmkde.DMKDE(feature_dim=256, sigma=0.8, contamination=0.05).fit(Xdup)
        fresh = r.normal(size=(2000, 6))
        over = np.mean(est.predict(fresh) == -1)
        est.calibrate(r.normal(size=(1000, 6)))
        cal = np.mean(est.predict(fresh) == -1)
    assert over > 0.3      # the leave-one-out score of a row still counts its twins
    assert cal < 0.1
    doc = dmkde.DMKDE.__doc__
    assert "duplicate" in doc and "calibrate" in doc
    assert "duplicate" in (ROOT / "README.md").read_text().lower()


# ---------------------------------------------------------------- B22 error types
def test_B22_documented_error_types_match_reality():
    est = fitted()
    sp = pytest.importorskip("scipy.sparse")
    cases = [
        (TypeError, lambda: est.score_samples(sp.csr_matrix(np.zeros((2, 3))))),
        (TypeError, lambda: est.score_samples({"a": 1})),
        (TypeError, lambda: est.partial_fit(X3[:2], 0.5)),
        (ValueError, lambda: est.score_samples([["a", "b", "c"]])),
        (ValueError, lambda: est.score_samples(np.zeros((2, 4)))),
        (ValueError, lambda: est.score_samples(np.full((2, 3), np.nan))),
        (ValueError, lambda: est.score_samples(np.zeros(3))),
        (ValueError, lambda: est.score_samples([[BIG, 1.0, 1.0]])),
        (dmkde.NotFittedError, lambda: dmkde.DMKDE().score_samples(X3)),
    ]
    for exc, call in cases:
        with pytest.raises(exc):
            call()
    readme = (ROOT / "README.md").read_text()
    assert "anything else raises `ValueError`" not in readme
    assert "a dict" in readme and "`TypeError`" in readme


# ---------------------------------------------------------------- B23 benchmarks
def test_B23_benchmark_timings_marked_with_their_version():
    text = (ROOT / "BENCHMARKS.md").read_text()
    assert "Timings were measured with 0.1.0" in text
