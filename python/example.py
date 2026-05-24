"""example.py — end-to-end Python usage of the dmkde package.

Run from the repo root after `pip install -e .`:

    python python/example.py
"""

from __future__ import annotations

import numpy as np

from dmkde import DMKDE, Mahalanobis, roc_auc


def make_bimodal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two well-separated 4-D Gaussian clusters; anomalies will live in the gap."""
    m1 = np.array([-1.8, 1.2, -1.8, 1.2])
    m2 = -m1
    z = rng.normal(size=(n, 4)) * 0.35
    mask = rng.random(size=(n, 1)) < 0.5
    return np.where(mask, z + m1, z + m2)


def main() -> None:
    rng = np.random.default_rng(42)

    X_train     = make_bimodal(400, rng)
    X_test_in   = make_bimodal(300, rng)
    X_test_out  = rng.normal(size=(300, 4)) * 0.4  # near origin = the gap between clusters

    model = DMKDE(feature_dim=256, sigma=1.5, random_state=42).fit(X_train)
    auc = roc_auc(model.score_samples(X_test_in), model.score_samples(X_test_out))
    print(f"DMKDE         AUC = {auc:.4f}   trace(ρ)={model.trace_:.4f}")

    base = Mahalanobis().fit(X_train)
    auc_b = roc_auc(base.score_samples(X_test_in), base.score_samples(X_test_out))
    print(f"Mahalanobis   AUC = {auc_b:.4f}")

    # Streaming: feed test-normal samples through the rank-1 EMA.
    model.partial_fit(X_test_in, alpha=0.005)
    auc_s = roc_auc(model.score_samples(X_test_in), model.score_samples(X_test_out))
    print(f"after stream  AUC = {auc_s:.4f}   ({model._impl.n_streamed} EMA updates)")

    # Threshold-based prediction: flag the bottom 5% of training as outliers.
    model.calibrate(X_train, contamination=0.05)
    preds = model.predict(np.vstack([X_test_in, X_test_out]))
    flagged = int((preds == -1).sum())
    print(f"predict() flagged {flagged}/{len(preds)} samples as anomalous "
          f"(threshold={model._default_threshold_:.4f})")


if __name__ == "__main__":
    main()
