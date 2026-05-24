"""dmkde — Density-matrix kernel density estimation for anomaly detection.

Production Python interface to the C++ core. The compiled module is
``dmkde._core``; this module wraps it with a scikit-learn-compatible
estimator API.

Algorithm references
--------------------
* Useche, D. H., González, F. A. et al. *Quantum density estimation with
  density matrices: Application to quantum anomaly detection.*
  Phys. Rev. A 109, 032418 (2024). arXiv:2201.10006
* AD-DMKDE, arXiv:2210.14796
* INQMAD (streaming), arXiv:2210.05061
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = [
    "DMKDE",
    "Mahalanobis",
    "roc_auc",
]


class DMKDE:
    """Density-matrix kernel density estimator with a scikit-learn-style API.

    Parameters
    ----------
    feature_dim : int
        Random Fourier Features embedding dimension `D`. Must be even.
        Larger `D` ⇒ better kernel approximation, more memory (O(D²) for ρ).
    sigma : float
        Bandwidth of the underlying Gaussian kernel
        ``k(x, y) = exp(-‖x - y‖² / 2σ²)``.
    random_state : int
        Seed for the RFF projection draws (reproducible across processes).

    Notes
    -----
    Higher ``score_samples`` ⇒ more "normal".
    ``decision_function`` follows the sklearn convention (higher ⇒ more
    inlier-like) and is equivalent to ``score_samples`` here.
    """

    def __init__(self, feature_dim: int = 256, sigma: float = 1.0,
                 random_state: int = 42) -> None:
        self.feature_dim = int(feature_dim)
        self.sigma = float(sigma)
        self.random_state = int(random_state)
        self._impl: _core.DMKDE | None = None

    def fit(self, X, y=None) -> "DMKDE":
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        self.n_features_in_ = X.shape[1]
        self._impl = _core.DMKDE(
            int(self.n_features_in_), self.feature_dim, self.sigma, self.random_state
        )
        self._impl.fit(X)
        return self

    def partial_fit(self, X, alpha: float = 0.01) -> "DMKDE":
        """Streaming rank-1 EMA update (INQMAD)."""
        if self._impl is None:
            raise RuntimeError("call fit() with an initial batch before partial_fit()")
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        for row in X:
            self._impl.update(row, float(alpha))
        return self

    def score_samples(self, X):
        if self._impl is None:
            raise RuntimeError("model not fitted")
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        return self._impl.score_batch(X)

    def decision_function(self, X):
        # Same sign convention as sklearn's OneClassSVM: higher = inlier.
        return self.score_samples(X)

    def predict(self, X, threshold: float | None = None):
        """Return +1 for inliers, -1 for outliers.

        If `threshold` is None, uses the 5th-percentile of training-set
        scores as a default (i.e. flags the most anomalous 5%).
        """
        scores = self.score_samples(X)
        if threshold is None:
            if not hasattr(self, "_default_threshold_"):
                raise RuntimeError(
                    "call .calibrate(X_train) once before predict() to set a default threshold"
                )
            threshold = self._default_threshold_
        return np.where(scores >= threshold, 1, -1)

    def calibrate(self, X_train, contamination: float = 0.05) -> "DMKDE":
        """Set the default predict() threshold from a holdout-set quantile."""
        s = self.score_samples(X_train)
        self._default_threshold_ = float(np.quantile(s, contamination))
        return self

    @property
    def trace_(self) -> float:
        if self._impl is None:
            return 0.0
        return float(self._impl.trace)


class Mahalanobis:
    """Mahalanobis-distance baseline matching the DMKDE sign convention.

    `score_samples` returns negative Mahalanobis distance, so larger values
    mean "more normal" — directly comparable for ROC-AUC.
    """

    def __init__(self) -> None:
        self._impl = _core.Mahalanobis()

    def fit(self, X, y=None) -> "Mahalanobis":
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        self._impl.fit(X)
        return self

    def score_samples(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        return self._impl.score_batch(X)

    def decision_function(self, X):
        return self.score_samples(X)


def roc_auc(normal_scores, anomaly_scores) -> float:
    """Mann–Whitney ROC-AUC; higher "normal" score is treated as positive."""
    return _core.roc_auc(
        np.ascontiguousarray(normal_scores, dtype=np.float64),
        np.ascontiguousarray(anomaly_scores, dtype=np.float64),
    )
