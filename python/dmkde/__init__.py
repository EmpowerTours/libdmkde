"""dmkde — Density-matrix kernel density estimation for anomaly detection.

Production Python interface to the C++ core. The compiled module is
``dmkde._core``; this module wraps it with a scikit-learn-compatible
estimator API (``fit`` / ``score_samples`` / ``decision_function`` /
``predict``, ``get_params`` / ``set_params``, pickling).

scikit-learn is optional. When it is importable, :class:`DMKDE` subclasses
``sklearn.base.BaseEstimator`` and ``OutlierMixin`` and validates input with
sklearn's own helpers; otherwise a minimal compatible fallback is used.

What the score is
-----------------
``score_samples(x) = φ(x)ᵀ ρ φ(x) = (1/N) Σᵢ ⟨φ(x), φ(xᵢ)⟩²``, which converges
to the mean of the *squared* Gaussian kernel ``exp(-‖x - xᵢ‖² / σ²)``. The
effective bandwidth is therefore ``σ/√2``, not ``σ``. The score is an
*unnormalized* density estimate in [0, 1]: it ranks points by density but
does not integrate to 1.

Algorithm references
--------------------
* D. H. Useche, O. A. Bustos-Brinez, J. A. Gallego-Mejia, F. A. González.
  *Quantum density estimation with density matrices: Application to quantum
  anomaly detection.* Phys. Rev. A 109, 032418 (2024). arXiv:2201.10006
* O. Bustos-Brinez, J. Gallego-Mejia, F. A. González. *AD-DMKDE: Anomaly
  Detection through Density Matrices and Fourier Features.* arXiv:2210.14796
  (2022)
* J. Gallego-Mejia, O. Bustos-Brinez, F. A. González. *InQMAD: Incremental
  Quantum Measurement Anomaly Detection.* arXiv:2210.05061 (2022)
"""

from __future__ import annotations

import inspect
import numbers

import numpy as np

from . import _core

__version__ = "0.2.0"

__all__ = [
    "DMKDE",
    "Mahalanobis",
    "NotFittedError",
    "roc_auc",
    "MAX_FEATURE_DIM",
]

#: Default cap on ``feature_dim`` (ρ needs ``feature_dim² × 8`` bytes).
MAX_FEATURE_DIM: int = int(_core.MAX_FEATURE_DIM)

_STATE_FORMAT = 1

# --------------------------------------------------------------------------
# Optional scikit-learn integration
# --------------------------------------------------------------------------
try:  # pragma: no cover - exercised in both configurations by the tests
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import OutlierMixin as _OutlierMixin
    from sklearn.exceptions import NotFittedError
    from sklearn.utils import check_random_state as _sk_check_random_state

    try:
        from sklearn.utils.validation import validate_data as _sk_validate_data
    except ImportError:  # scikit-learn < 1.6
        _sk_validate_data = None
    _HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAVE_SKLEARN = False
    _sk_validate_data = None
    _sk_check_random_state = None

    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator is used before ``fit``."""

    class _BaseEstimator:
        """Minimal stand-in for ``sklearn.base.BaseEstimator``."""

        @classmethod
        def _get_param_names(cls):
            sig = inspect.signature(cls.__init__)
            return sorted(p.name for p in sig.parameters.values()
                          if p.name != "self" and p.kind != p.VAR_KEYWORD)

        def get_params(self, deep=True):
            return {k: getattr(self, k) for k in self._get_param_names()}

        def set_params(self, **params):
            valid = self._get_param_names()
            for k, v in params.items():
                if k not in valid:
                    raise ValueError(f"Invalid parameter {k!r} for estimator {self!r}. "
                                     f"Valid parameters are: {valid!r}.")
                setattr(self, k, v)
            return self

        def __repr__(self):
            args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
            return f"{type(self).__name__}({args})"

    class _OutlierMixin:
        """Minimal stand-in for ``sklearn.base.OutlierMixin``."""

        _estimator_type = "outlier_detector"

        def fit_predict(self, X, y=None, **kwargs):
            return self.fit(X, **kwargs).predict(X)


def _fallback_feature_names(X):
    """Column names of a DataFrame-like X, as sklearn's ``_get_feature_names``.

    Returns an object array of the names if they are all strings, None if X
    has no columns or none of them is a string, and raises TypeError for a
    mix of string and non-string names.
    """
    cols = getattr(X, "columns", None)
    if cols is None:
        return None
    names = np.asarray(list(cols), dtype=object)
    is_str = np.array([isinstance(c, str) for c in names], dtype=bool)
    if names.size == 0 or not is_str.any():
        return None
    if not is_str.all():
        types = sorted({type(c).__qualname__ for c in names})
        raise TypeError(
            "Feature names are only supported if all input features have string "
            f"names, but your input has {types} as feature name / column name types.")
    return names


def _fallback_check_feature_names(est, X, reset: bool) -> None:
    """Record ``feature_names_in_`` on fit and enforce it later, as sklearn does."""
    names = _fallback_feature_names(X)
    if reset:
        if names is not None:
            est.feature_names_in_ = names
        else:
            est.__dict__.pop("feature_names_in_", None)
        return
    fitted = getattr(est, "feature_names_in_", None)
    cls = type(est).__name__
    if names is None and fitted is None:
        return
    import warnings

    if names is not None and fitted is None:
        warnings.warn(f"X has feature names, but {cls} was fitted without feature names",
                      UserWarning, stacklevel=5)
        return
    if names is None:
        warnings.warn(f"X does not have valid feature names, but {cls} was fitted with "
                      "feature names", UserWarning, stacklevel=5)
        return
    if len(names) != len(fitted) or np.any(names != fitted):
        unexpected = sorted(set(names) - set(fitted))
        missing = sorted(set(fitted) - set(names))
        msg = "The feature names should match those that were passed during fit.\n"
        if unexpected:
            msg += f"Feature names unseen at fit time:\n- {unexpected}\n"
        if missing:
            msg += f"Feature names seen at fit time, yet now missing:\n- {missing}\n"
        if not unexpected and not missing:
            msg += "Feature names must be in the same order as they were in fit.\n"
        raise ValueError(msg)


def _fallback_validate(est, X, reset: bool) -> np.ndarray:
    """sklearn-style validation used when scikit-learn is not installed."""
    if hasattr(X, "toarray") or type(X).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse data was passed, but dense data is required.")
    _fallback_check_feature_names(est, X, reset)
    X = np.asarray(X)
    if X.dtype.kind == "c":
        raise ValueError("Complex data not supported")
    try:
        X = np.ascontiguousarray(X, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"Unable to convert array of dtype {X.dtype} to float64") from e
    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got {X.ndim}D array instead. Reshape your data "
            "either using array.reshape(-1, 1) if your data has a single feature "
            "or array.reshape(1, -1) if it contains a single sample.")
    if X.shape[0] < 1:
        raise ValueError(f"Found array with 0 sample(s) (shape={X.shape}) while a "
                         "minimum of 1 is required.")
    if X.shape[1] < 1:
        raise ValueError(f"Found array with 0 feature(s) (shape={X.shape}) while a "
                         "minimum of 1 is required.")
    if np.isnan(X).any():
        raise ValueError("Input X contains NaN.")
    if not np.isfinite(X).all():
        raise ValueError("Input X contains infinity or a value too large for dtype('float64').")
    if reset:
        est.n_features_in_ = X.shape[1]
    elif X.shape[1] != est.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features, but {type(est).__name__} is "
                         f"expecting {est.n_features_in_} features as input.")
    return X


def _validate(est, X, reset: bool) -> np.ndarray:
    """Return X as a finite, C-contiguous float64 (n_samples, n_features) array.

    Enforces ``n_features_in_`` (set when ``reset`` is True) and raises
    sklearn-style ``ValueError`` for 1-D input, empty input, 0 features,
    NaN and inf.
    """
    try:
        if _sk_validate_data is not None:
            X = _sk_validate_data(est, X, reset=reset, dtype=np.float64, order="C")
        elif _HAVE_SKLEARN and hasattr(est, "_validate_data"):  # pragma: no cover
            X = est._validate_data(X, reset=reset, dtype=np.float64, order="C")
        else:
            X = _fallback_validate(est, X, reset)
    except OverflowError as e:
        # e.g. the Python int 10**400, which numpy cannot cast to float64.
        raise ValueError(f"Input X contains a value too large for float64: {e}") from e
    return np.ascontiguousarray(X, dtype=np.float64)


# Attributes that input validation with reset=True may set or delete.
_VALIDATION_ATTRS = ("n_features_in_", "feature_names_in_")


def _fit_transaction(est, build):
    """Run ``build(X_validated)`` so that a failure leaves ``est`` unchanged.

    ``_validate(reset=True)`` sets ``n_features_in_`` (and sklearn's
    ``feature_names_in_``) before the C++ core has accepted the data. If
    validation or ``build`` raises, those attributes are restored to their
    previous values, so the previous fitted state (or the unfitted state)
    stays intact. ``build`` must not touch ``est``; it returns the new fitted
    attributes, which are assigned only after it succeeded.
    """
    saved = {k: est.__dict__[k] for k in _VALIDATION_ATTRS if k in est.__dict__}
    try:
        return build()
    except BaseException:
        for k in _VALIDATION_ATTRS:
            est.__dict__.pop(k, None)
        est.__dict__.update(saved)
        raise


def _check_fitted(est, attr: str = "_impl") -> None:
    if not hasattr(est, attr):
        raise NotFittedError(
            f"This {type(est).__name__} instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator.")


def _resolve_seed(random_state) -> int:
    """Map random_state (None | int | RandomState | Generator) to a uint64 seed.

    An int is used as the seed directly: since 0.2.0 it gives the same RFF
    weights on every platform (0.1.x weights were platform-dependent, and a
    seed builds different weights than it did in 0.1.x). None draws from
    NumPy's global RNG.
    """
    if isinstance(random_state, (bool, np.bool_)):
        raise ValueError(f"random_state must be None, an int, a RandomState or a "
                         f"Generator, got {random_state!r}")
    if isinstance(random_state, numbers.Integral):
        seed = int(random_state)
        if not 0 <= seed < 2**64:
            raise ValueError(f"random_state must be in [0, 2**64), got {seed}")
        return seed
    if random_state is None:
        rs = (_sk_check_random_state(None) if _sk_check_random_state is not None
              else np.random.mtrand._rand)
        return int(rs.randint(0, 2**63 - 1, dtype=np.int64))
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, 2**63 - 1, dtype=np.int64))
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 2**63 - 1))
    raise ValueError(f"random_state must be None, an int, a RandomState or a "
                     f"Generator, got {random_state!r}")


def _loo_training_scores(scores: np.ndarray, n: int) -> np.ndarray:
    """Exact leave-one-out scores of the ``n`` training rows of a batch fit.

    ``scores[i] = φᵢᵀ ρ φᵢ`` with ``ρ = (1/n) Σⱼ φⱼ φⱼᵀ`` contains the row's own
    term ``(φᵢ·φᵢ)² / n``. Every RFF embedding has unit norm
    (``‖φ‖² = (2/D) Σₖ (cos² + sin²) = (2/D)(D/2) = 1``), so that term is
    exactly ``1/n`` and

        scores[i] = (1/n) [1 + Σ_{j≠i} (φᵢ·φⱼ)²]

    The model refit without row i is ``ρ₋ᵢ = (1/(n-1)) Σ_{j≠i} φⱼ φⱼᵀ``, whose
    score for row i is

        φᵢᵀ ρ₋ᵢ φᵢ = (1/(n-1)) Σ_{j≠i} (φᵢ·φⱼ)² = (n·scores[i] - 1) / (n - 1).

    This is (to O(1/n)) what a fresh point from the training distribution
    sees, so it is what the threshold is taken from; the in-sample scores
    sit about ``1/n`` too high and flag far more than ``contamination`` of
    new data when scores are small (high d, small σ). The result is clipped
    to ``<= scores`` (true in exact arithmetic) against rounding. For
    ``n = 1`` the leave-one-out model is empty and scores every point 0, so
    0.0 is returned.
    """
    if n < 2:
        return np.zeros_like(scores)
    # Mathematically loo <= s (as s <= 1). When every row is the same point,
    # s rounds to 1 ± a few ulp and (n·s - 1)/(n - 1) can round an ulp above
    # s, which put offset_ above every training score and flagged all of them.
    return np.minimum((n * scores - 1.0) / (n - 1), scores)


def _conformal_rank(n: int, c: float) -> int:
    """k = floor((n + 1)·c), the order statistic used as the threshold.

    The relative 1e-12 guards against (n + 1)·c landing a rounding error
    below an integer, e.g. 0.07·100 = 7.000000000000001 but 0.29·100 =
    28.999999999999996.
    """
    return int(np.floor((n + 1) * c * (1.0 + 1e-12)))


def _order_statistic_threshold(scores: np.ndarray, c: float, what: str,
                               stacklevel: int = 3, raise_if_too_few: bool = False) -> float:
    """Threshold flagging at most a fraction ``c`` of fresh data, on average.

    Derivation (split-conformal / rank argument). Let s_1..s_n be the scores
    of n points and s_0 the score of a fresh point, all n + 1 exchangeable
    (same distribution, any order equally likely) and without ties. The rank
    of s_0 among the n + 1 values is then uniform on {1, ..., n + 1}, so for
    the k-th smallest of s_1..s_n, s_(k),

        P(s_0 < s_(k)) = P(rank of s_0 <= k) = k / (n + 1).

    Flagging ``score < offset_`` with ``offset_ = s_(k)`` and
    ``k = floor((n + 1)·c)`` therefore flags an expected fraction
    ``k/(n + 1) <= c`` of fresh data, for every n (ties only lower it).
    ``np.quantile(s, c)`` instead interpolates at position ``(n - 1)·c``,
    which for small n sits above s_(k) and flags more than c: at c = 0.05
    0.37-0.44 of fresh data for n = 2, 0.17-0.31 for n = 5 and 0.07-0.13
    for n = 20. For large n the two agree (k/(n + 1) -> c from below).

    Held-out scores (``calibrate``) are exactly exchangeable with fresh
    ones. The leave-one-out training scores used by ``fit`` are each a mean
    kernel to the n - 1 other rows, while a fresh point averages over all n,
    so they are exchangeable only approximately (to O(1/n)); the bound is
    checked by Monte Carlo in the tests.

    If k = 0, i.e. n + 1 < 1/c, even the smallest score would flag
    1/(n + 1) > c of fresh data, so no threshold can keep the rate at c:
    ``0.0`` is returned (scores are >= 0, so nothing is flagged) with a
    warning, or, with ``raise_if_too_few``, ValueError is raised. When n·c < 5 the realised rate of one fit is noisy (it is
    k/(n + 1) only on average over training sets) and a warning says so.
    """
    import warnings

    n = int(scores.size)
    k = _conformal_rank(n, c)
    if k < 1:
        inv = 1.0 / c
        need = f"at least {int(np.ceil(inv)) - 1}" if np.isfinite(inv) else "more than 1e308"
        if raise_if_too_few:
            raise ValueError(
                f"{what}: {n} sample(s) are too few to set a threshold at contamination="
                f"{c:g} (needs {need}): even the lowest score would flag about 1/{n + 1} of "
                "new normal data. offset_ is unchanged.")
        warnings.warn(
            f"{what}: {n} sample(s) are too few to set a threshold at contamination="
            f"{c:g} (needs {need}): even the lowest score would flag about "
            f"1/{n + 1} of new normal data. offset_ = 0.0, so predict() flags nothing. "
            "Use more data or call calibrate() on more held-out normal data.",
            UserWarning, stacklevel=stacklevel)
        return 0.0
    if n * c < 5:
        warnings.warn(
            f"{what}: only n_samples*contamination = {n * c:.3g} expected outliers "
            f"among {n} samples, so the threshold is noisy: on average it flags "
            f"{k}/{n + 1} = {k / (n + 1):.3g} <= {c:g} of new normal data, but a "
            "single fit can flag much more or less.",
            UserWarning, stacklevel=stacklevel)
    return float(np.partition(scores, k - 1)[k - 1])


# A median leave-one-out training score below this many multiples of 1/D
# means the scores are dominated by the RFF noise floor (see _warn_noise_floor).
_NOISE_FLOOR_FACTOR = 3.0


def _warn_noise_floor(loo: np.ndarray, feature_dim: int, sigma: float, d: int,
                      stacklevel: int = 3) -> None:
    """Warn when training scores sit near the RFF noise floor of ~1/D.

    For two points far apart relative to σ, ``⟨φ(x), φ(y)⟩`` is a sum of D/2
    cosines with effectively random phases: mean ≈ 0, variance ≈ 1/D. Its
    square, which the score averages, therefore has mean ≈ 1/D whatever the
    true kernel value is. When the median leave-one-out training score is
    below ``_NOISE_FLOOR_FACTOR / D`` the kernel signal is at most about
    twice that noise and the ranking degrades towards chance. Measured
    against the exact squared-kernel density (d = 2-100 standardized
    Gaussian features, D = 256 and 1024): at N = 500 every setting with
    median·D <= 1.01 had AUC ≈ 0.5 (0.45-0.56), median·D 1.1-1.5 lost
    0.07-0.18 AUC, 1.5-2.1 lost 0.01-0.02, 3-4 lost at most 0.02 and
    above 10 at most 0.006. With few training rows the
    threshold is not sufficient: at N = 10 fits with median·D 3-6 (no
    warning) still lost up to 0.09 AUC.
    """
    if loo.size < 2:
        return
    med = float(np.median(loo))
    if med * feature_dim < _NOISE_FLOOR_FACTOR:
        import warnings

        warnings.warn(
            f"DMKDE training scores are near the random-feature noise floor: median "
            f"leave-one-out score {med:.3g} is only {med * feature_dim:.2g} x "
            f"1/feature_dim = {1.0 / feature_dim:.3g}, the score two unrelated points "
            "get from random-feature noise alone. The kernel signal between typical "
            f"training points (sigma={sigma:g}, {d} features) is at most comparable "
            "to that noise, so the ranking degrades towards random. Either sigma is "
            "too small for the data (standardize the features and increase sigma: a "
            "rule of thumb is sigma ≈ sqrt(n_features), or, after standardizing, the "
            "median pairwise distance / sqrt(2)) or feature_dim is too small for the "
            "noise floor 1/feature_dim to be negligible.",
            UserWarning, stacklevel=stacklevel)


# 1 - (median leave-one-out score) below these means the scores are saturated
# near 1 (see _warn_saturated).
_SATURATION_LEVEL = 0.02
_ROUNDING_LEVEL = 1e-9


def _warn_saturated(loo: np.ndarray, sigma: float, stacklevel: int = 3) -> None:
    """Warn when the training scores are saturated near 1 (σ too large).

    When σ is large compared with the spread of the data, every pair of
    points has kernel value close to 1 and ``1 - s(x)`` is about the mean
    squared distance from x to the training rows divided by σ²: the score is
    then a distance-to-the-mean detector and no longer models multimodal or
    non-convex structure. Measured (N = 500, D = 512, two clusters at ±2 with
    anomalies between them, d = 2 and 8): AUC 0.999 at σ = √d, where
    1 - median ≈ 0.66, but 0.0003 or less from 1 - median ≈ 0.08 down, i.e.
    the gap anomalies score as the most normal points. On unimodal data the
    ranking survives (AUC unchanged from 1 - median = 0.2 down to 2e-11)
    until rounding: from 1 - median ≈ 1e-12 scores start to tie, and at
    σ ≈ 1e8·√d every score is exactly 1 and nothing is detected. Hence one
    warning below ``_SATURATION_LEVEL`` and a stronger one below
    ``_ROUNDING_LEVEL``.
    """
    if loo.size < 2:
        return
    gap = 1.0 - float(np.median(loo))
    if gap >= _SATURATION_LEVEL:
        return
    import warnings

    msg = (f"DMKDE training scores are saturated near 1: 1 - median leave-one-out score "
           f"= {gap:.3g}. sigma={sigma:g} is large compared with the spread of the training "
           "data (or the rows are nearly identical), so the kernel is almost flat over it: "
           "the score reduces to the squared distance to the training data and no longer "
           "models multimodal or non-convex structure (anomalies between two clusters score "
           "as normal).")
    if gap < _ROUNDING_LEVEL:
        msg += (" At this level the scores are within rounding of 1, so they tie and "
                "detection is lost (predict() flags little or nothing).")
    msg += " Standardize the features and use a smaller sigma (rule of thumb sqrt(n_features))."
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


def _check_alpha(alpha) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, numbers.Real):
        raise ValueError(f"alpha must be a float in (0, 1], got {alpha!r}")
    a = float(alpha)
    if not 0.0 < a <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    return a


def _check_y_is_not_alpha(y, X) -> None:
    """Reject a ``y`` that is really a positional ``alpha``.

    ``partial_fit(X, y=None, alpha=...)`` ignores ``y``. 0.1.1 caught only a
    real scalar passed as ``y``; a 0-d array (``np.array(0.5)``) or a short
    list (``[0.5]``) was silently ignored and the default alpha used. Now any
    0-d ``y``, and any ``y`` whose length differs from the number of rows
    of X, raises TypeError.
    """
    if y is None:
        return
    msg = "partial_fit(X, y=None, alpha=...): pass alpha by keyword; "
    if np.ndim(y) == 0:
        raise TypeError(msg + f"a scalar was given as y ({y!r})")
    try:
        n_y, n_x = np.shape(y)[0], np.shape(X)[0]
    except (TypeError, ValueError, IndexError):  # X is validated (and rejected) later
        return
    if n_y != n_x:
        raise TypeError(msg + f"y has {n_y} entries but X has {n_x} rows (y is ignored)")


def _check_contamination(c) -> float:
    if isinstance(c, (bool, np.bool_)) or not isinstance(c, numbers.Real):
        raise ValueError(f"contamination must be a float in (0, 0.5], got {c!r}")
    c = float(c)
    if not 0.0 < c <= 0.5:
        raise ValueError(f"contamination must be in (0, 0.5], got {c}")
    return c


class DMKDE(_OutlierMixin, _BaseEstimator):
    """Density-matrix kernel density estimator with a scikit-learn API.

    Parameters
    ----------
    feature_dim : int, default=256
        Random Fourier Features embedding dimension ``D``; positive and even.
        Memory is O(D²) for ρ, so ``D > MAX_FEATURE_DIM`` (2**14, i.e. 2 GiB)
        is refused unless ``allow_large_feature_dim=True``.
    sigma : float, default=1.0
        Bandwidth of the RFF kernel ``k(x, y) = exp(-‖x - y‖² / 2σ²)``;
        finite and > 0. The score averages ``k²``, so the *effective*
        bandwidth of the density estimate is ``σ/√2``. **The default suits
        only a few standardized features.** Standardize the features first
        (e.g. ``StandardScaler``) and let σ grow with the number of
        features d: a rule of thumb is ``σ ≈ sqrt(d)``, or the median
        pairwise distance of the *standardized* training data divided by
        ``sqrt(2)``. If σ
        is too small every pair of points is far apart in kernel units, the
        scores collapse to the random-feature noise floor of about
        ``1/feature_dim`` and the ranking is close to random; ``fit`` emits
        a ``UserWarning`` when that happens.
    random_state : None, int, RandomState or Generator, default=42
        Seed for the RFF projection draws. An int is used directly as the
        C++ seed and gives the same weights on every OS (since 0.2.0; a
        seed builds a different model than in 0.1.x); None draws a seed
        from NumPy's global RNG.
    contamination : float, default=0.05
        Target false-alarm rate on fresh normal data, in (0, 0.5]. ``fit``
        sets ``offset_`` to the k-th smallest of the n *leave-one-out*
        training scores, ``k = floor((n + 1)·contamination)``, so that on
        average about this fraction or less of fresh data from the training
        distribution is flagged (exact for held-out scores in ``calibrate``;
        approximate, to O(1/n), for the leave-one-out scores of ``fit``): ``k/(n + 1)``, e.g.
        0.04995 for n = 1000 at 0.05. With few samples the rate is lower than
        ``contamination`` on average and noisy from fit to fit (``fit``
        warns when ``n·contamination < 5``); with ``n + 1 < 1/contamination``
        (n < 19 at 0.05) nothing is flagged and ``fit`` warns. (Each
        in-sample score contains its own row's term, exactly
        ``1/n_samples``, so in-sample scores run higher than fresh ones.)
    allow_large_feature_dim : bool, default=False
        Opt in to ``feature_dim > MAX_FEATURE_DIM``, and to more than 2**28
        random-feature weights ``(feature_dim/2)·n_features`` (2 GiB; e.g.
        more than 2**21 features at ``feature_dim=256``).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in ``fit``; enforced on every later call.
    offset_ : float
        Threshold on ``score_samples``: ``decision_function = score_samples
        - offset_``. Set by ``fit`` (from ``contamination``, the order
        statistic above on the exact leave-one-out training scores
        ``(n·s - 1)/(n - 1)``) or by ``calibrate`` (the same rule on held-out
        scores). When there are too few samples for the rule (including a
        single training sample) ``offset_ = 0.0``: nothing is flagged, and
        a ``UserWarning`` says so.

    Notes
    -----
    * ``score_samples`` is the Born-rule score ``φ(x)ᵀ ρ φ(x)`` in [0, 1],
      an *unnormalized* density estimate; higher ⇒ more normal.
    * The estimator is a *novelty* detector: ``offset_`` is calibrated for
      new data. ``predict`` on the training set itself (and so
      ``fit_predict``) flags fewer than ``contamination`` of it, because
      each training row's own score includes its own ``1/n_samples`` term.
      ``fit_predict`` is deliberately ``fit(X).predict(X)``: scikit-learn
      requires the two to agree, so thresholding the leave-one-out training
      scores there instead would still fail ``check_outliers_fit_predict``.
    * Random-feature noise floor: for points far apart relative to σ the
      score is about ``1/feature_dim`` regardless of the true kernel, so
      only scores well above ``1/feature_dim`` carry ranking information.
    * ``decision_function`` follows the sklearn outlier-detector convention:
      negative ⇒ outlier, and ``predict(X) == where(decision_function(X) >= 0,
      1, -1)``. (In 0.1.0 it returned the raw score.)
    * σ too large: when every training score is close to 1 (``fit`` warns
      below 1 - median = 0.02) the score is effectively a distance to the
      training mean and loses multimodal structure; below 1e-9 scores tie at
      1 and nothing is detected.
    * Duplicate training rows: a row's leave-one-out score still contains
      its duplicate copies (kernel value 1 each), which fresh points do not
      have, so with many duplicates and a small σ ``offset_`` is too high and
      ``predict`` over-flags new data (measured: 76% instead of 5% with every
      row three times). Deduplicate (``np.unique(X, axis=0)``) or
      ``calibrate`` on held-out data.
    * NaN / inf, complex and non-numeric inputs raise ``ValueError``
      (sparse input and objects NumPy cannot read as numbers raise
      ``TypeError``, as in scikit-learn); inputs must have
      ``n_features_in_`` columns and be 2-D.
    """

    def __init__(self, feature_dim: int = 256, sigma: float = 1.0,
                 random_state=42, contamination: float = 0.05,
                 allow_large_feature_dim: bool = False) -> None:
        self.feature_dim = feature_dim
        self.sigma = sigma
        self.random_state = random_state
        self.contamination = contamination
        self.allow_large_feature_dim = allow_large_feature_dim

    # ------------------------------------------------------------------ fit
    def _check_params(self):
        fd = self.feature_dim
        if isinstance(fd, (bool, np.bool_)) or not isinstance(fd, numbers.Integral):
            raise ValueError(f"feature_dim must be a positive even int, got {fd!r}")
        fd = int(fd)
        if fd <= 0 or fd % 2:
            raise ValueError(f"feature_dim must be a positive even int, got {fd}")
        if fd > MAX_FEATURE_DIM and not self.allow_large_feature_dim:
            raise ValueError(
                f"feature_dim={fd} exceeds MAX_FEATURE_DIM={MAX_FEATURE_DIM} "
                f"(rho would need {fd * fd * 8 / 2**30:.1f} GiB); pass "
                "allow_large_feature_dim=True to opt in")
        sg = self.sigma
        if isinstance(sg, (bool, np.bool_)) or not isinstance(sg, numbers.Real):
            raise ValueError(f"sigma must be a finite float > 0, got {sg!r}")
        sg = float(sg)
        if not (np.isfinite(sg) and sg > 0 and np.isfinite(1.0 / sg)):
            raise ValueError(f"sigma must be a finite float > 0, got {sg}")
        c = _check_contamination(self.contamination)
        return fd, sg, c

    def fit(self, X, y=None) -> "DMKDE":
        """Fit ρ on X and set ``offset_`` from ``contamination``.

        ``offset_`` is the k-th smallest exact leave-one-out training score,
        ``k = floor((n + 1)·contamination)`` (see the class docstring). Re-fitting
        replaces the model and any threshold set by ``calibrate``. Warns
        (``UserWarning``) when the training scores sit at the random-feature
        noise floor, i.e. σ is too small for the data.
        """
        fd, sg, c = self._check_params()

        def build():
            Xv = _validate(self, X, reset=True)
            if (not self.allow_large_feature_dim
                    and (fd // 2) * Xv.shape[1] > int(_core.MAX_WEIGHT_COUNT)):
                raise ValueError(
                    f"feature_dim={fd} with {Xv.shape[1]} features needs "
                    f"{(fd // 2) * Xv.shape[1]} random-feature weights "
                    f"({(fd // 2) * Xv.shape[1] * 8 / 2**30:.1f} GiB), above the default cap "
                    f"of {int(_core.MAX_WEIGHT_COUNT)} (2 GiB); pass "
                    "allow_large_feature_dim=True to opt in")
            seed = _resolve_seed(self.random_state)
            impl = _core.DMKDE(int(Xv.shape[1]), fd, sg, seed,
                               bool(self.allow_large_feature_dim))
            impl.fit(Xv)
            n = Xv.shape[0]
            loo = _loo_training_scores(impl.score_batch(Xv), n)
            if n < 2:
                import warnings

                warnings.warn(
                    "DMKDE was fitted on a single sample: there is no data to estimate "
                    "a threshold from, so offset_ = 0.0 and predict() flags nothing. "
                    "Call calibrate() on held-out normal data to set a threshold.",
                    UserWarning, stacklevel=4)
                return impl, 0.0, loo
            # stacklevel: helper -> build -> _fit_transaction -> fit -> caller.
            offset = _order_statistic_threshold(loo, c, "DMKDE.fit", stacklevel=5)
            # The data-quality warnings are emitted here, before anything is
            # committed: under ``-W error`` they raise inside the transaction
            # and the previous fit (or the unfitted state) is kept (0.1.1
            # warned after installing the new model).
            _warn_noise_floor(loo, fd, sg, int(Xv.shape[1]), stacklevel=5)
            _warn_saturated(loo, sg, stacklevel=5)
            return impl, offset, loo

        # On failure (including a warning turned into an error) the previous
        # fit, or the unfitted state, is kept intact.
        self._impl, self.offset_, _ = _fit_transaction(self, build)
        return self

    def partial_fit(self, X, y=None, alpha: float = 0.01) -> "DMKDE":
        """Streaming rank-1 EMA update ``ρ ← (1-α)ρ + α φ(x)φ(x)ᵀ`` per row.

        On an unfitted estimator the first call is equivalent to ``fit(X)``.
        ``alpha`` must be passed by keyword (the second positional argument is
        ``y``, per the sklearn API). ``offset_`` is not changed; call
        ``calibrate`` to refresh it. Updates are unconditional: rows are
        absorbed even if they are anomalous.

        Forgetting is fast: after k updates the original fit keeps weight
        ``(1-α)^k``, so the effective memory is about ``1/α`` rows. With the
        default ``α = 0.01`` a single 1000-row batch leaves the original fit
        a weight of ``0.99**1000 ≈ 4e-5``, i.e. it is erased and ρ describes
        only the streamed rows.

        The batch is atomic: every row is validated first (finite input and
        finite RFF embedding), so if any row is rejected the call raises and
        ρ and ``n_streamed`` are unchanged.
        """
        _check_y_is_not_alpha(y, X)
        a = _check_alpha(alpha)  # also before the first-call fit (0.1.1 skipped it)
        if not hasattr(self, "_impl"):
            return self.fit(X)
        X = _validate(self, X, reset=False)
        self._impl.update_batch(X, a)
        return self

    # -------------------------------------------------------------- scoring
    def score_samples(self, X) -> np.ndarray:
        """Born-rule score per row (higher ⇒ more normal), shape (n_samples,)."""
        _check_fitted(self)
        X = _validate(self, X, reset=False)
        return self._impl.score_batch(X)

    def decision_function(self, X) -> np.ndarray:
        """``score_samples(X) - offset_``; negative ⇒ outlier."""
        _check_fitted(self)
        return self.score_samples(X) - self.offset_

    def predict(self, X, threshold: float | None = None) -> np.ndarray:
        """Return +1 for inliers and -1 for outliers.

        With ``threshold=None`` (default) a row is an outlier when
        ``decision_function(X) < 0``, i.e. its score is below ``offset_``.
        ``offset_`` is set by ``fit`` from the leave-one-out training scores
        (see the class docstring), or by ``calibrate``. An explicit
        ``threshold`` (a finite real number; anything else raises
        ``ValueError``) is compared with ``score_samples`` instead.
        """
        if threshold is None:
            d = self.decision_function(X)
        else:
            if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, numbers.Real):
                raise ValueError(f"threshold must be a finite real number or None, got "
                                 f"{threshold!r}")
            t = float(threshold)
            if not np.isfinite(t):
                # NaN made every comparison False (all flagged) and ±inf flags
                # everything or nothing; neither is a threshold.
                raise ValueError(f"threshold must be finite, got {threshold!r}")
            d = self.score_samples(X) - t
        return np.where(d >= 0, 1, -1)

    def calibrate(self, X, contamination: float | None = None) -> "DMKDE":
        """Set ``offset_`` from the scores of held-out normal data ``X``.

        ``offset_`` is the k-th smallest of the n scores ``score_samples(X)``
        with ``k = floor((n + 1)·contamination)``: for data exchangeable with
        ``X`` this flags an expected fraction ``k/(n + 1) <= contamination``
        (the same rule ``fit`` applies to the leave-one-out training scores;
        see the class docstring). If ``k = 0`` (n + 1 < 1/contamination,
        e.g. fewer than 19 rows at 0.05) no threshold can meet the target:
        ``ValueError`` is raised and ``offset_`` is left unchanged (0.1.1 set
        ``offset_ = 0.0`` with a warning, which silently switched detection
        off). ``fit`` still only warns in that case, because it must return a
        fitted estimator; ``calibrate`` exists only to set the threshold, so
        it refuses instead of failing open. A warning says the rate is noisy
        when ``n·contamination < 5``.

        ``contamination`` defaults to the estimator's ``contamination``
        parameter and must be in (0, 0.5]. Use held-out data: the in-sample
        scores of the rows ρ was fitted on are biased upwards (by exactly
        ``1/n_samples`` per row after a batch fit), so calibrating on the
        training set flags too much new data.
        """
        c = _check_contamination(self.contamination if contamination is None
                                 else contamination)
        self.offset_ = _order_statistic_threshold(self.score_samples(X), c,
                                                  "DMKDE.calibrate", stacklevel=3,
                                                  raise_if_too_few=True)
        return self

    @property
    def trace_(self) -> float:
        """trace(ρ) (1 for a fitted model; 0.0 before fit)."""
        if not hasattr(self, "_impl"):
            return 0.0
        return float(self._impl.trace)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_impl")

    # -------------------------------------------------------------- pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        impl = state.pop("_impl", None)
        if impl is not None:
            state["_core_state"] = impl.get_state()
        state["_dmkde_state_format"] = _STATE_FORMAT
        return state

    def __setstate__(self, state):
        state = dict(state)
        fmt = state.pop("_dmkde_state_format", None)
        if fmt != _STATE_FORMAT:
            raise ValueError(f"unsupported DMKDE pickle format {fmt!r}")
        core_state = state.pop("_core_state", None)
        state.pop("_sklearn_version", None)
        impl = None
        if core_state is not None:
            impl = _core.DMKDE.from_state(core_state)
            _check_pickled_n_features(state, impl.input_dim, "DMKDE")
            off = state.get("offset_", None)
            if (isinstance(off, (bool, np.bool_)) or not isinstance(off, numbers.Real)
                    or not (0.0 <= float(off) <= 1.0 + 1e-9)):
                raise ValueError(f"DMKDE pickle: offset_ must be a finite float in [0, 1] for a "
                                 f"fitted model, got {off!r}")
            state["offset_"] = float(off)
        self.__dict__.update(state)
        if impl is not None:
            self._impl = impl


def _check_pickled_n_features(state: dict, input_dim: int, cls: str) -> None:
    nf = state.get("n_features_in_", None)
    if (isinstance(nf, (bool, np.bool_)) or not isinstance(nf, numbers.Integral)
            or int(nf) != input_dim):
        raise ValueError(f"{cls} pickle: n_features_in_ must be the model's input_dim "
                         f"{input_dim}, got {nf!r}")


class Mahalanobis(_BaseEstimator):
    """Mahalanobis-distance baseline matching the DMKDE sign convention.

    ``score_samples`` returns the negative Mahalanobis distance, so larger
    values mean "more normal" — directly comparable for ROC-AUC. Needs at
    least 2 finite training rows.

    * The covariance ridge is per-feature relative, ``C + 1e-6 · diag(C)``,
      so for **non-constant** features the score is invariant to rescaling
      any feature independently (its units).
    * **Constant features** (every training value identical, min == max)
      have no scale to be relative to and use an **absolute**
      one: their inverse variance is fixed at ``1e6`` (as 0.1.0's
      ``1e-6 · I`` ridge gave), independent of every other column. A
      deviation δ from the constant value adds ``1e6·δ²`` to the squared
      distance (distance ``1000·|δ|`` on its own), so any change of a
      feature that never varied in training is far out. This part is *not*
      unit-invariant: a deviation of 0.01 scores -10 in one unit and -10⁴
      after multiplying that column by 1000. Data where every feature is
      constant is rejected.
    * Features with a standard deviation below ~1e-154 or above ~1e154
      cannot have their inverse variance represented in double precision
      and are rejected (``ValueError`` naming the feature); rescale them.
    * The mean is stored as a double in the column's own units, so it is
      rounded to that column's precision. For a column whose spread is only
      a few ulp of its magnitude (e.g. values 1.7e18 and 1.7e18 + 1 ulp) this
      rounding is comparable to the spread itself and distances along that
      column are off by up to ~0.5·ulp/std; subtract a reference value
      (e.g. the first timestamp) from such columns first.

    * A finite query whose distance exceeds ``DBL_MAX`` (~1.8e308; e.g.
      ``x - mean`` near ±1e308 along a low-variance direction) cannot be
      represented: ``score_samples`` raises ``ValueError`` ("not finite")
      instead of returning ``-inf``. Smaller but overflowing quadratic forms
      are recomputed on the rescaled difference and give a large finite
      distance.

    There is no threshold: ``decision_function`` returns ``score_samples``
    unchanged (always <= 0, with no ``offset_``), unlike ``DMKDE`` whose
    ``decision_function`` is ``score_samples - offset_``, and there is no
    ``predict``. Threshold the scores yourself, e.g. at a quantile of the
    scores of held-out normal data.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X, y=None) -> "Mahalanobis":
        # On failure the previous fit (or the unfitted state) is kept intact.
        self._impl = _fit_transaction(self, lambda: self._build(X))
        return self

    def _build(self, X):
        X = _validate(self, X, reset=True)
        if X.shape[0] < 2:
            raise ValueError(f"Mahalanobis needs at least 2 samples, got "
                             f"n_samples={X.shape[0]}")
        impl = _core.Mahalanobis()
        impl.fit(X)
        return impl

    def score_samples(self, X) -> np.ndarray:
        _check_fitted(self)
        X = _validate(self, X, reset=False)
        return self._impl.score_batch(X)

    def decision_function(self, X) -> np.ndarray:
        """Same as ``score_samples`` (negative distance, <= 0); no offset."""
        return self.score_samples(X)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_impl")

    def __getstate__(self):
        state = self.__dict__.copy()
        impl = state.pop("_impl", None)
        if impl is not None:
            state["_core_state"] = impl.get_state()
        return state

    def __setstate__(self, state):
        state = dict(state)
        core_state = state.pop("_core_state", None)
        state.pop("_sklearn_version", None)
        impl = None
        if core_state is not None:
            impl = _core.Mahalanobis.from_state(core_state)
            _check_pickled_n_features(state, impl.input_dim, "Mahalanobis")
        self.__dict__.update(state)
        if impl is not None:
            self._impl = impl


def roc_auc(normal_scores, anomaly_scores) -> float:
    """Mann–Whitney ROC-AUC; higher "normal" score is treated as positive.

    Both inputs must be non-empty 1-D arrays of finite scores; otherwise a
    ``ValueError`` is raised (as scikit-learn's ``roc_auc_score`` does).
    """
    out = []
    for name, s in (("normal_scores", normal_scores), ("anomaly_scores", anomaly_scores)):
        a = np.asarray(s)
        if np.iscomplexobj(a) or (a.dtype == object and any(isinstance(v, complex)
                                                           for v in a.ravel())):
            raise ValueError(f"{name}: complex data is not supported")
        try:
            a = a.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"{name} cannot be converted to float64: {e}") from e
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
        if a.size == 0:
            raise ValueError(f"{name} is empty; ROC-AUC is undefined")
        if np.isnan(a).any():
            raise ValueError(f"Input {name} contains NaN.")
        if not np.isfinite(a).all():
            raise ValueError(f"Input {name} contains infinity.")
        out.append(np.ascontiguousarray(a))
    return float(_core.roc_auc(out[0], out[1]))
