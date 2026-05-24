"""benchmark_kdd.py — DMKDE on KDDCup'99 SA-subset network intrusion data.

Dataset: KDDCup'99 SMTP+HTTP subset (~100k flows after 10%-sample),
38 numeric features (drop 3 categorical), 3.4% intrusion (anomaly) rate.
Reference: Lippmann et al., DARPA Intrusion Detection Evaluation (1999).
Fetched via sklearn.datasets.fetch_kddcup99.

Anomaly-detection setup mirrors benchmark_creditcard.py:
    Train each detector on a random sample of NORMAL flows only.
    Test on a held-out mix of normal + all anomalies.
    Report ROC-AUC and PR-AUC.

Run from repo root:
    python python/benchmark_kdd.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from dmkde import DMKDE, Mahalanobis


def load_data():
    print("Fetching KDDCup'99 SA subset (10% sample)...")
    data = fetch_kddcup99(subset="SA", percent10=True, random_state=42)
    df = pd.DataFrame(data.data)
    # Drop categorical columns: 1=protocol_type, 2=service, 3=flag
    keep = [c for c in df.columns if c not in (1, 2, 3)]
    X = df[keep].to_numpy(dtype=np.float64)
    y = np.array([0 if v == b"normal." else 1 for v in data.target])
    print(f"  shape={X.shape}  anomaly={int(y.sum())}/{len(y)} "
          f"({y.mean() * 100:.2f}%)")
    return X, y


def split(X, y, n_train, n_test_normal, seed):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    anom_idx   = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    n_test_normal = min(n_test_normal, len(normal_idx) - n_train)
    return X[normal_idx[:n_train]], \
           X[normal_idx[n_train : n_train + n_test_normal]], \
           X[anom_idx]


def run_detector(name, fit_fn, score_fn, train, testN, testA):
    t0 = time.perf_counter()
    model = fit_fn(train)
    t1 = time.perf_counter()
    sN = score_fn(model, testN)
    sA = score_fn(model, testA)
    t2 = time.perf_counter()
    y_true  = np.concatenate([np.zeros(len(sN)), np.ones(len(sA))])
    y_score = -np.concatenate([sN, sA])
    return {
        "name":   name,
        "roc":    roc_auc_score(y_true, y_score),
        "pr":     average_precision_score(y_true, y_score),
        "fit_s":  t1 - t0,
        "inf_s":  t2 - t1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train",       type=int,   default=10_000)
    ap.add_argument("--n_test_normal", type=int,   default=10_000)
    ap.add_argument("--feature_dim",   type=int,   default=1024)
    ap.add_argument("--sigma",         type=float, default=5.0)
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    X, y = load_data()
    train, testN, testA = split(X, y, args.n_train, args.n_test_normal, args.seed)
    scaler = StandardScaler().fit(train)
    train, testN, testA = scaler.transform(train), scaler.transform(testN), scaler.transform(testA)
    print(f"\nTrain {train.shape}  TestNormal {testN.shape}  TestAnom {testA.shape}")
    print(f"DMKDE: feature_dim={args.feature_dim} σ={args.sigma}\n")

    results = []
    results.append(run_detector(
        f"DMKDE σ={args.sigma} D={args.feature_dim}",
        fit_fn=lambda X: DMKDE(feature_dim=args.feature_dim, sigma=args.sigma,
                                random_state=args.seed).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA,
    ))
    results.append(run_detector("Mahalanobis",
        fit_fn=lambda X: Mahalanobis().fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA))
    results.append(run_detector("IsolationForest",
        fit_fn=lambda X: IsolationForest(n_estimators=200, contamination="auto",
                                          random_state=args.seed).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA))
    results.append(run_detector("LOF",
        fit_fn=lambda X: LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA))
    results.append(run_detector("OneClassSVM",
        fit_fn=lambda X: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X),
        score_fn=lambda m, X: m.score_samples(X),
        train=train, testN=testN, testA=testA))

    results.sort(key=lambda r: -r["roc"])

    print(f"{'detector':<26}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'fit (s)':>8}  {'inf (s)':>8}")
    for r in results:
        print(f"  {r['name']:<24}  {r['roc']:>8.4f}  {r['pr']:>8.4f}  "
              f"{r['fit_s']:>8.2f}  {r['inf_s']:>8.2f}")


if __name__ == "__main__":
    main()
