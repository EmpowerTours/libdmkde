"""dmkde.pyod — PyOD-compatible adapter for DMKDE.

PyOD's BaseDetector convention is:
    decision_function(X)  →  higher = MORE anomalous
    labels_                →  0 = inlier, 1 = outlier
This is the OPPOSITE of the native DMKDE ``score_samples`` convention, where
higher means more normal. The adapter negates ``score_samples`` once.

Usage:
    from dmkde.pyod import DMKDEDetector
    det = DMKDEDetector(feature_dim=256, sigma=1.5, contamination=0.05)
    det.fit(X_train)
    labels = det.predict(X_test)        # 0 = inlier, 1 = outlier
    proba  = det.predict_proba(X_test)  # (n, 2) PyOD outlier probabilities

Requires `pyod` installed. The import is delayed until ``DMKDEDetector`` is
first accessed, so importing ``dmkde.pyod`` does not pull pyod in. The class
is built once and cached at module level, so ``isinstance`` checks and
pickling work.
"""

from __future__ import annotations

from . import DMKDE


def _require_pyod():
    try:
        from pyod.models.base import BaseDetector
    except ImportError as e:
        raise ImportError(
            "DMKDEDetector requires pyod. Install with `pip install pyod`."
        ) from e
    return BaseDetector


def _build_class():
    Base = _require_pyod()
    from sklearn.utils.validation import check_is_fitted

    class DMKDEDetector(Base):
        """PyOD-compatible DMKDE anomaly detector.

        Parameters
        ----------
        feature_dim : int, default=256
            Random Fourier Features dimension D.
        sigma : float, default=1.0
            RFF kernel bandwidth (effective density bandwidth σ/√2).
        contamination : float, default=0.1
            Expected proportion of outliers, in (0, 0.5]; sets
            ``threshold_`` used by ``predict``.
        random_state : None, int, RandomState or Generator, default=42
            RNG seed for the RFF projections.
        """

        def __init__(self, feature_dim=256, sigma=1.0,
                     contamination=0.1, random_state=42):
            super().__init__(contamination=contamination)
            self.feature_dim = feature_dim
            self.sigma = sigma
            self.random_state = random_state

        def fit(self, X, y=None):
            model = DMKDE(
                feature_dim=self.feature_dim,
                sigma=self.sigma,
                random_state=self.random_state,
                contamination=self.contamination,
            ).fit(X)
            self._model = model
            self.n_features_in_ = model.n_features_in_
            self._set_n_classes(y)
            # PyOD stores per-sample anomaly scores from training.
            # Negate native scores so "higher = more anomalous" holds.
            self.decision_scores_ = -model.score_samples(X)
            self._process_decision_scores()
            return self

        def decision_function(self, X):
            check_is_fitted(self, ["_model", "decision_scores_"])
            return -self._model.score_samples(X)

    DMKDEDetector.__module__ = __name__
    DMKDEDetector.__qualname__ = "DMKDEDetector"
    return DMKDEDetector


def __getattr__(name: str):
    """Build DMKDEDetector on first access and cache it in the module."""
    if name == "DMKDEDetector":
        cls = _build_class()
        globals()["DMKDEDetector"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
