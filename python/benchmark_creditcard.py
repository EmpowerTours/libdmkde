"""benchmark_creditcard.py — DMKDE on the Kaggle Credit Card Fraud dataset.

Dataset: 284,807 European credit-card transactions, 492 fraudulent
(0.173% positive class). Features are 28 PCA components + Time + Amount.
Reference: Dal Pozzolo et al., "Calibrating Probability with Undersampling
for Unbalanced Classification" (2015). Fetched via OpenML id 1597.

Anomaly-detection setup:
    Train each detector on a random sample of NORMAL transactions only.
    Test on a held-out mix of normal + all fraud cases.
    Report ROC-AUC and PR-AUC; PR-AUC is the relevant metric under 0.17%
    base rate.

Run from repo root after first ensuring sklearn + DMKDE installed:
    python python/benchmark_creditcard.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from dmkde import DMKDE, Mahalanobis


def load_data(seed: int):
    print("Fetching CreditCardFraudDetection (OpenML id 1597)...")
    ds = fetch_openml("CreditCardFraudDetection", version=1, parser="liac-arff",
                      as_frame=False)
    X = np.asarray(ds.data, dtype=np.float64)
    y = np.asarray(ds.target).astype(int)
    print(f"  shape={X.shape}  fraud={int((y == 1).sum())}/{len(y)} "
          f"({(y == 1).mean() * 100:.3f}%)")
    return X, y


def split(X, y, n_train, n_test_normal, seed):
    """Train on n_train sampled normals; test on n_test_normal new normals + all fraud."""
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    fraud_idx  = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    train_idx       = normal_idx[:n_train]
    test_norm_idx   = normal_idx[n_train : n_train + n_test_normal]
    return X[train_idx], X[test_norm_idx], X[fraud_idx]


def run_detector(name, fit_fn, score_fn, train, testN, testA):
    t0 = time.perf_counter()
    model = fit_fn(train)
    t1 = time.perf_counter()
    sN = score_fn(model, testN)
    sA = score_fn(model, testA)
    t2 = time.perf_counter()
    y_true  = np.concatenate([np.zeros(len(sN)), np.ones(len(sA))])
    y_score = -np.concatenate([sN, sA])   # invert because higher = more normal
    auc     = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)
    print(f"  {name:<16}  ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
          f"fit={(t1 - t0):.2f}s  inf={(t2 - t1):.2f}s")
    return {"roc_auc": auc, "pr_auc": pr_auc,
            "fit_s": t1 - t0, "inf_s": t2 - t1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train",       type=int, default=10_000,
                    help="number of NORMAL samples used for training")
    ap.add_argument("--n_test_normal", type=int, default=10_000,
                    help="number of NORMAL samples in test")
    ap.add_argument("--feature_dim",   type=int, default=512)
    ap.add_argument("--sigma",         type=float, default=2.0)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    X, y = load_data(args.seed)
    train, testN, testA = split(X, y, args.n_train, args.n_test_normal, args.seed)

    scaler = StandardScaler().fit(train)
    train, testN, testA = scaler.transform(train), scaler.transform(testN), scaler.transform(testA)
    print(f"\nTrain {train.shape}   TestNormal {testN.shape}   TestFraud {testA.shape}")
    print(f"DMKDE: feature_dim={args.feature_dim} σ={args.sigma}\n")

    results = {}
    print(f"{'detector':<16}  ROC/PR AUC + timing")

    results["DMKDE"] = run_detector(
        "DMKDE",
        fit_fn=lambda X: DMKDE(feature_dim=args.feature_dim, sigma=args.sigma,
                                random_state=args.seed).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA,
    )

    results["Mahalanobis"] = run_detector(
        "Mahalanobis",
        fit_fn=lambda X: Mahalanobis().fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA,
    )

    results["IsolationForest"] = run_detector(
        "IsolationForest",
        fit_fn=lambda X: IsolationForest(
            n_estimators=200, contamination="auto", random_state=args.seed
        ).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA,
    )

    # LOF with novelty=True for inference on unseen points
    results["LOF"] = run_detector(
        "LOF",
        fit_fn=lambda X: LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA,
    )

    # OneClassSVM is O(N²) — keep training small or skip if N > 10k
    if args.n_train <= 10_000:
        results["OneClassSVM"] = run_detector(
            "OneClassSVM",
            fit_fn=lambda X: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X),
            score_fn=lambda m, X: m.score_samples(X),
            train=train, testN=testN, testA=testA,
        )
    else:
        print("  OneClassSVM     skipped (O(N²) — too slow for n_train > 10k)")

    print("\n=== SUMMARY (Kaggle Credit Card Fraud, 0.17% positive rate) ===")
    print(f"{'detector':<16}  {'ROC-AUC':>8}  {'PR-AUC':>8}")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["roc_auc"]):
        print(f"  {name:<14}  {r['roc_auc']:>8.4f}  {r['pr_auc']:>8.4f}")


if __name__ == "__main__":
    main()
