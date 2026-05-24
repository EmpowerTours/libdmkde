"""benchmark.py — DMKDE vs scikit-learn baselines on standard scenarios.

Compares the Born-rule density-matrix detector against:
  - Isolation Forest        (Liu et al., 2008)
  - One-Class SVM (RBF)     (Schölkopf et al., 2001)
  - Local Outlier Factor    (Breunig et al., 2000)
  - Mahalanobis distance    (linear baseline)

Across four scenarios:
  1. Gaussian normal        — Mahalanobis-optimal case
  2. Bimodal manifold       — anomalies in the gap between two clusters
  3. Ring                   — anomalies in the hub of a hypersphere
  4. Sklearn make_classification — standard imbalanced classification
                              repurposed as one-class anomaly detection

Run from repo root:
    python python/benchmark.py
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from dmkde import DMKDE, Mahalanobis


# ============================================================
#  Scenario generators
# ============================================================
def gen_gaussian(d, n_train, n_test_norm, n_test_anom, rng):
    mu = 0.5 * np.cos(np.arange(d) * 0.7)
    train = rng.normal(size=(n_train, d)) * 0.6 + mu
    testN = rng.normal(size=(n_test_norm, d)) * 0.6 + mu
    # anomalies far from mean
    testA = []
    while len(testA) < n_test_anom:
        x = rng.uniform(-4, 4, d)
        if np.linalg.norm(x - mu) > np.sqrt(d * 2.5):
            testA.append(x)
    return train, testN, np.array(testA)


def gen_bimodal(d, n_train, n_test_norm, n_test_anom, rng):
    mu1 = np.where(np.arange(d) % 2 == 0, -1.8, 1.2)
    mu2 = -mu1

    def sample(n):
        coin = rng.random(n) < 0.5
        z = rng.normal(size=(n, d)) * 0.35
        return np.where(coin[:, None], z + mu1, z + mu2)

    train = sample(n_train)
    testN = sample(n_test_norm)
    testA = rng.normal(size=(n_test_anom, d)) * 0.4  # near origin = the gap
    return train, testN, testA


def gen_ring(d, n_train, n_test_norm, n_test_anom, rng):
    def on_ring(n):
        x = rng.normal(size=(n, d))
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        r = rng.normal(loc=3.0, scale=0.18, size=(n, 1))
        return x / norm * np.maximum(r, 0.5)

    train = on_ring(n_train)
    testN = on_ring(n_test_norm)
    testA = rng.normal(size=(n_test_anom, d)) * 0.4  # hub of the ring
    return train, testN, testA


def gen_sklearn(d, n_train, n_test_norm, n_test_anom, rng):
    # sklearn make_classification, then we relabel class 0 as "normal", class 1 as anomaly
    total = n_train + n_test_norm + n_test_anom
    X, y = make_classification(
        n_samples=total * 2,
        n_features=d,
        n_informative=min(d, 6),
        n_redundant=0,
        n_classes=2,
        weights=[0.95, 0.05],
        class_sep=1.2,
        random_state=int(rng.integers(0, 2**30)),
    )
    Xn = X[y == 0]
    Xa = X[y == 1]
    rng.shuffle(Xn)
    rng.shuffle(Xa)
    train = Xn[:n_train]
    testN = Xn[n_train : n_train + n_test_norm]
    testA = Xa[:n_test_anom]
    return train, testN, testA


SCENARIOS: dict[str, Callable] = {
    "gaussian": gen_gaussian,
    "bimodal":  gen_bimodal,
    "ring":     gen_ring,
    "sklearn":  gen_sklearn,
}


# ============================================================
#  Detector adapters: every detector returns "higher = more normal"
# ============================================================
def _wrap_dmkde(feature_dim, sigma, seed):
    return lambda: ("DMKDE", DMKDE(feature_dim=feature_dim, sigma=sigma, random_state=seed))


def _wrap_maha():
    def make():
        return "Mahalanobis", Mahalanobis()
    return make


class _SklearnAdapter:
    """Wraps a sklearn detector so that score_samples returns 'higher = more normal'."""

    def __init__(self, name, est):
        self.name = name
        self.est = est

    def fit(self, X):
        if isinstance(self.est, LocalOutlierFactor):
            # LOF needs novelty=True to enable predict on new data
            self.est.set_params(novelty=True).fit(X)
        else:
            self.est.fit(X)
        return self

    def score_samples(self, X):
        # IsolationForest.score_samples: higher = more normal ✓
        # OneClassSVM.score_samples: higher = more normal ✓
        # LocalOutlierFactor.score_samples: higher = more normal ✓
        return self.est.score_samples(X)


def _wrap_iforest(seed):
    return lambda: ("IsolationForest", _SklearnAdapter(
        "IsolationForest",
        IsolationForest(n_estimators=200, contamination="auto", random_state=seed),
    ))


def _wrap_ocsvm():
    return lambda: ("OneClassSVM", _SklearnAdapter(
        "OneClassSVM",
        OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
    ))


def _wrap_lof():
    return lambda: ("LOF", _SklearnAdapter(
        "LOF",
        LocalOutlierFactor(n_neighbors=20, novelty=True),
    ))


# ============================================================
#  Driver
# ============================================================
def run_scenario(name, gen, d, n_train, n_test, seed,
                 feature_dim=256, sigma=1.5):
    rng = np.random.default_rng(seed)
    train, testN, testA = gen(d, n_train, n_test, n_test, rng)

    detector_factories = [
        _wrap_dmkde(feature_dim, sigma, seed),
        _wrap_maha(),
        _wrap_iforest(seed),
        _wrap_ocsvm(),
        _wrap_lof(),
    ]

    print(f"\n=== {name.upper()}  (d={d}, train={len(train)}, test={len(testN)}+{len(testA)}) ===")
    print(f"{'detector':<18}  {'AUC':>7}  {'fit (ms)':>10}  {'inf (ms)':>10}")
    results = {}
    for factory in detector_factories:
        det_name, det = factory()
        t0 = time.perf_counter()
        det.fit(train)
        t1 = time.perf_counter()
        sN = det.score_samples(testN)
        sA = det.score_samples(testA)
        t2 = time.perf_counter()
        y_true = np.concatenate([np.ones(len(sN)), np.zeros(len(sA))])
        y_score = np.concatenate([sN, sA])
        auc = roc_auc_score(y_true, y_score)
        print(f"  {det_name:<16}  {auc:>7.4f}  {(t1 - t0) * 1000:>10.1f}  "
              f"{(t2 - t1) * 1000:>10.1f}")
        results[det_name] = auc
    return results


def main():
    ap = argparse.ArgumentParser(description="DMKDE vs sklearn anomaly-detection baselines")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--feature_dim", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    args = ap.parse_args()

    print(f"libdmkde benchmark — DMKDE vs sklearn detectors")
    print(f"d={args.d}  feature_dim={args.feature_dim}  σ={args.sigma}  "
          f"n_train={args.n_train}  n_test={args.n_test}  seed={args.seed}")

    all_results = {}
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            print(f"unknown scenario: {scenario}; skipping")
            continue
        all_results[scenario] = run_scenario(
            scenario, SCENARIOS[scenario],
            args.d, args.n_train, args.n_test, args.seed,
            feature_dim=args.feature_dim, sigma=args.sigma,
        )

    # summary table
    print("\n=== AUC SUMMARY ===")
    detectors = ["DMKDE", "Mahalanobis", "IsolationForest", "OneClassSVM", "LOF"]
    header = f"{'scenario':<10}  " + "  ".join(f"{d:>16}" for d in detectors)
    print(header)
    print("-" * len(header))
    for sc, res in all_results.items():
        line = f"{sc:<10}  " + "  ".join(
            f"{res.get(d, float('nan')):>16.4f}" for d in detectors
        )
        print(line)


if __name__ == "__main__":
    main()
