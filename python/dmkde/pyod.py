"""dmkde.pyod — PyOD-compatible adapter for DMKDE.

PyOD's BaseDetector convention is:
    decision_function(X)  →  higher = MORE anomalous
    labels_                →  0 = inlier, 1 = outlier
This is the OPPOSITE of sklearn / our native DMKDE convention, where
higher means more normal. The adapter inverts the sign once.

Usage:
    from dmkde.pyod import DMKDEDetector
    det = DMKDEDetector(feature_dim=256, sigma=1.5, contamination=0.05)
    det.fit(X_train)
    labels = det.predict(X_test)  # 0 = inlier, 1 = outlier

Requires `pyod` installed. The import is delayed so non-PyOD users are
not forced to install it.
"""

from __future__ import annotations

import numpy as np

from . import DMKDE


def _require_pyod():
    try:
        from pyod.models.base import BaseDetector
    except ImportError as e:
        raise ImportError(
            "DMKDEDetector requires pyod. Install with `pip install pyod`."
        ) from e
    return BaseDetector


def __getattr__(name: str):
    """Lazily construct DMKDEDetector so importing dmkde.pyod doesn't pull pyod in."""
    if name == "DMKDEDetector":
        Base = _require_pyod()

        class DMKDEDetector(Base):
            """PyOD-compatible DMKDE anomaly detector.

            Parameters
            ----------
            feature_dim : int, default=256
                Random Fourier Features dimension D.
            sigma : float, default=1.0
                Gaussian-kernel bandwidth.
            contamination : float, default=0.1
                Expected proportion of outliers; sets the threshold used
                by `predict`. Lower ⇒ stricter.
            random_state : int, default=42
                RNG seed for the RFF projections.
            """

            def __init__(self, feature_dim=256, sigma=1.0,
                         contamination=0.1, random_state=42):
                super().__init__(contamination=contamination)
                self.feature_dim = int(feature_dim)
                self.sigma = float(sigma)
                self.random_state = int(random_state)

            def fit(self, X, y=None):
                X = np.ascontiguousarray(X, dtype=np.float64)
                if X.ndim != 2:
                    raise ValueError("X must be 2-D")
                self.n_features_in_ = X.shape[1]
                self._model = DMKDE(
                    feature_dim=self.feature_dim,
                    sigma=self.sigma,
                    random_state=self.random_state,
                ).fit(X)
                # PyOD stores per-sample anomaly scores from training.
                # Negate native scores so "higher = more anomalous" holds.
                self.decision_scores_ = -self._model.score_samples(X)
                self._process_decision_scores()
                return self

            def decision_function(self, X):
                X = np.ascontiguousarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X[None, :]
                return -self._model.score_samples(X)

        return DMKDEDetector
    raise AttributeError(name)
