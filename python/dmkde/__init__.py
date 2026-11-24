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

__version__ = "0.1.1"

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


def _fallback_validate(est, X, reset: bool) -> np.ndarray:
    """sklearn-style validation used when scikit-learn is not installed."""
    if hasattr(X, "toarray") or type(X).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse data was passed, but dense data is required.")
    X = np.asarray(X)
    if X.dtype.kind == "c":
        raise ValueError("Complex data not supported")
    try:
        X = np.ascontiguousarray(X, dtype=np.float64)
    except (TypeError, ValueError) as e:
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
    if _sk_validate_data is not None:
        X = _sk_validate_data(est, X, reset=reset, dtype=np.float64, order="C")
    elif _HAVE_SKLEARN and hasattr(est, "_validate_data"):  # pragma: no cover
        X = est._validate_data(X, reset=reset, dtype=np.float64, order="C")
    else:
        X = _fallback_validate(est, X, reset)
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

    An int is used as the seed directly (so ``random_state=42`` reproduces
    the 0.1.0 projections exactly). None draws from NumPy's global RNG.
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
        bandwidth of the density estimate is ``σ/√2``.
    random_state : None, int, RandomState or Generator, default=42
        Seed for the RFF projection draws. An int is used directly as the
        C++ seed; None draws a seed from NumPy's global RNG.
    contamination : float, default=0.05
        Expected outlier fraction in the training data, in (0, 0.5].
        ``fit`` sets ``offset_`` to this quantile of the training scores.
    allow_large_feature_dim : bool, default=False
        Opt in to ``feature_dim > MAX_FEATURE_DIM``.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen in ``fit``; enforced on every later call.
    offset_ : float
        Threshold on ``score_samples``: ``decision_function = score_samples
        - offset_``. Set by ``fit`` (from ``contamination``) or ``calibrate``.

    Notes
    -----
    * ``score_samples`` is the Born-rule score ``φ(x)ᵀ ρ φ(x)`` in [0, 1],
      an *unnormalized* density estimate; higher ⇒ more normal.
    * ``decision_function`` follows the sklearn outlier-detector convention:
      negative ⇒ outlier, and ``predict(X) == where(decision_function(X) >= 0,
      1, -1)``. (In 0.1.0 it returned the raw score.)
    * NaN / inf inputs raise ``ValueError``; inputs must have
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

        Re-fitting replaces the model and any threshold set by ``calibrate``.
        """
        fd, sg, c = self._check_params()

        def build():
            Xv = _validate(self, X, reset=True)
            seed = _resolve_seed(self.random_state)
            impl = _core.DMKDE(int(Xv.shape[1]), fd, sg, seed,
                               bool(self.allow_large_feature_dim))
            impl.fit(Xv)
            return impl, float(np.quantile(impl.score_batch(Xv), c))

        # On failure the previous fit (or the unfitted state) is kept intact.
        self._impl, self.offset_ = _fit_transaction(self, build)
        return self

    def partial_fit(self, X, y=None, alpha: float = 0.01) -> "DMKDE":
        """Streaming rank-1 EMA update ``ρ ← (1-α)ρ + α φ(x)φ(x)ᵀ`` per row.

        On an unfitted estimator the first call is equivalent to ``fit(X)``.
        ``alpha`` must be passed by keyword (the second positional argument is
        ``y``, per the sklearn API). ``offset_`` is not changed; call
        ``calibrate`` to refresh it. Updates are unconditional: rows are
        absorbed even if they are anomalous.

        The batch is atomic: every row is validated first (finite input and
        finite RFF embedding), so if any row is rejected the call raises and
        ρ and ``n_streamed`` are unchanged.
        """
        if y is not None and np.ndim(y) == 0 and isinstance(y, numbers.Real) \
                and not isinstance(y, (bool, np.bool_)):
            raise TypeError("partial_fit(X, y=None, alpha=...): pass alpha by keyword; "
                            "a scalar was given as y")
        if not hasattr(self, "_impl"):
            return self.fit(X)
        X = _validate(self, X, reset=False)
        a = float(alpha)
        if not 0.0 < a <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
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
        ``offset_`` is set by ``fit`` to the ``contamination`` quantile of the
        training scores, or by ``calibrate``. An explicit ``threshold`` is
        compared with ``score_samples`` instead.
        """
        if threshold is None:
            d = self.decision_function(X)
        else:
            d = self.score_samples(X) - float(threshold)
        return np.where(d >= 0, 1, -1)

    def calibrate(self, X, contamination: float | None = None) -> "DMKDE":
        """Set ``offset_`` to the ``contamination`` quantile of ``score_samples(X)``.

        ``contamination`` defaults to the estimator's ``contamination``
        parameter and must be in (0, 0.5].
        """
        c = _check_contamination(self.contamination if contamination is None
                                 else contamination)
        self.offset_ = float(np.quantile(self.score_samples(X), c))
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
        self.__dict__.update(state)
        if core_state is not None:
            self._impl = _core.DMKDE.from_state(core_state)


class Mahalanobis(_BaseEstimator):
    """Mahalanobis-distance baseline matching the DMKDE sign convention.

    ``score_samples`` returns the negative Mahalanobis distance, so larger
    values mean "more normal" — directly comparable for ROC-AUC. Needs at
    least 2 finite training rows. The covariance ridge is
    ``1e-6 · trace(C)/d``, which keeps the score invariant to rescaling.
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
        self.__dict__.update(state)
        if core_state is not None:
            self._impl = _core.Mahalanobis.from_state(core_state)


def roc_auc(normal_scores, anomaly_scores) -> float:
    """Mann–Whitney ROC-AUC; higher "normal" score is treated as positive.

    Both inputs must be non-empty 1-D arrays of finite scores; otherwise a
    ``ValueError`` is raised (as scikit-learn's ``roc_auc_score`` does).
    """
    out = []
    for name, s in (("normal_scores", normal_scores), ("anomaly_scores", anomaly_scores)):
        a = np.asarray(s, dtype=np.float64)
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
