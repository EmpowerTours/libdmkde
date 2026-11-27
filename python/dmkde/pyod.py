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

from . import DMKDE, _loo_training_scores, _validate


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
            The default is PyOD's own default for every detector, kept on
            purpose so ``DMKDEDetector`` is a drop-in PyOD model; it differs
            from the native ``DMKDE.contamination`` default of 0.05. Pass it
            explicitly to get the same threshold from both.
            Target false-alarm rate on fresh normal data, in (0, 0.5];
            sets ``threshold_`` used by ``predict``. ``decision_scores_``
            are the exact leave-one-out training scores, negated, and
            ``threshold_ = -DMKDE.offset_`` (the k-th smallest leave-one-out
            score, ``k = floor((N + 1)·contamination)``, rather than PyOD's
            percentile), so ``predict`` flags on average at most (for large
            N about) this fraction of fresh data from the training
            distribution. ``labels_`` marks the training rows above
            ``threshold_``: ``k - 1`` of them, slightly under
            ``contamination·N``.
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
            # PyOD stores per-sample anomaly scores from training and derives
            # threshold_ / labels_ from them. Use the exact leave-one-out
            # training scores (as PyOD's KNN / LOF exclude each point from its
            # own neighbourhood): the in-sample score contains the row's own
            # 1/n term, which would put threshold_ too low for new data.
            # Negate so "higher = more anomalous" holds.
            Xv = _validate(model, X, reset=False)
            self.decision_scores_ = -_loo_training_scores(
                model._impl.score_batch(Xv), Xv.shape[0])
            self._process_decision_scores()
            # PyOD's threshold_ is np.percentile of the training scores,
            # whose interpolation over-flags fresh data for small N (at
            # contamination 0.05: ~0.4 at N = 2, 0.07-0.13 at N = 20). Use
            # the same order statistic as DMKDE.fit instead: predict flags
            # ``-s > threshold_``, i.e. ``s < model.offset_``, the k-th
            # smallest leave-one-out score with k = floor((N + 1)·c), which
            # flags at most a fraction c of fresh data on average (see
            # dmkde._order_statistic_threshold). When N is too small for
            # that, offset_ = 0 and nothing is flagged (DMKDE.fit warned).
            self.threshold_ = -float(model.offset_)
            self.labels_ = (self.decision_scores_ > self.threshold_).astype("int").ravel()
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
