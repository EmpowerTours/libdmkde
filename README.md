# libdmkde

A production-grade C++17 implementation of **Density-Matrix Kernel Density
Estimation** for anomaly detection, with a streaming O(1)-per-sample update.

The algorithm is **not new** — it was introduced by Useche, González, et al.
in *Phys. Rev. A* (2024). What this repository provides is the engineering
that has been missing from the published research code: a single-header C++
library, a deterministic benchmark harness, reproducible numbers against a
classical baseline, and an MIT-licensed build that a production team can
adopt without untangling Jupyter notebooks.

## Citation — please cite the original authors

If you use this library in academic work, **cite the papers, not this repo:**

```bibtex
@article{useche2024quantum,
  author  = {Useche, Diego H. and Gonz{\'a}lez, Fabio A. and others},
  title   = {Quantum density estimation with density matrices:
             Application to quantum anomaly detection},
  journal = {Physical Review A},
  volume  = {109},
  pages   = {032418},
  year    = {2024},
  doi     = {10.1103/PhysRevA.109.032418}
}

@misc{useche2022inqmad,
  author = {Useche, Diego H. and others},
  title  = {INQMAD: Incremental Quantum Measurement Anomaly Detection},
  year   = {2022},
  note   = {arXiv:2210.05061}
}

@misc{gonzalez2022addmkde,
  author = {Gonz{\'a}lez, Fabio A. and others},
  title  = {AD-DMKDE: Anomaly Detection through Density Matrices
            and Fourier Features},
  year   = {2022},
  note   = {arXiv:2210.14796}
}
```

The reference Python notebooks live at
[`diegour1/QuantumAnomalyDetection`](https://github.com/diegour1/QuantumAnomalyDetection),
[`diegour1/QDEMDE`](https://github.com/diegour1/QDEMDE), and
[`Joaggi/lean-dmkde`](https://github.com/Joaggi/lean-dmkde).
The general probabilistic-DL primitives are at
[`fagonzalezo/kdm`](https://github.com/fagonzalezo/kdm).

## Algorithm in one paragraph

Each input `x ∈ R^d` is embedded into `R^D` using Random Fourier Features
(Rahimi & Recht, 2007), which approximates a Gaussian kernel
`k(x, y) = exp(-‖x - y‖² / 2σ²)` as an inner product `⟨φ(x), φ(y)⟩`.
Training builds the empirical density operator

    ρ = (1/N) Σᵢ φ(xᵢ) φ(xᵢ)ᵀ      (D × D, PSD, trace 1)

and the Born-rule score

    s(x) = φ(x)ᵀ ρ φ(x)             ∈ [0, 1]

estimates the density of `x` under the training distribution. Low `s` ⇒
anomalous. The streaming variant updates `ρ` via a rank-1 EMA:

    ρ ← (1 - α) ρ + α · φ(x) φ(x)ᵀ

with `O(D²)` time and `O(1)` extra memory per sample.

## Why this exists

The published research code is high-quality but not production-ready: all
public implementations are Jupyter notebooks with single-digit star counts,
none are on PyPI, none have a stable public API, and none have a streaming
update path. As of May 2026:

- No package on PyPI, conda-forge, crates.io, or npm matches.
- No major quantum library (Qiskit Machine Learning, PennyLane, TensorFlow
  Quantum, Cirq) exposes a density-matrix anomaly detector.
- PyOD's 60+ detectors include nothing density-matrix based.
- Closed commercial offerings (e.g. Multiverse Singularity) use different
  algorithm families.

This library closes that gap for the C++ side of the stack.

## Build

### C++ (header-only)

```sh
make           # builds benchmark + tests
make test      # runs the unit tests
make bench     # runs the reproducible benchmark
```

Requires `g++` or `clang++` with C++17. No external dependencies.

### Python bindings

```sh
pip install .            # builds the C++ extension and installs the dmkde package
python python/example.py # runs the end-to-end demo
```

Requires Python ≥ 3.9, `pybind11`, and `numpy`. Build uses `setuptools` +
`Pybind11Extension`; no system-wide installs needed.

## Use

### C++

```cpp
#include "dmkde.hpp"

// 4 input dims, 256-D RFF embedding, Gaussian kernel bandwidth σ = 1.5.
dmkde::DMKDE model(/*input_dim=*/4, /*feature_dim=*/256, /*sigma=*/1.5);

std::vector<std::vector<double>> train = /* normal samples */;
model.fit(train);

double s = model.score(new_point.data());        // higher = more "normal"
model.update(drift_point.data(), /*alpha=*/0.01); // streaming EMA update
```

### Python (sklearn-style)

```python
import numpy as np
from dmkde import DMKDE, Mahalanobis, roc_auc

model = DMKDE(feature_dim=256, sigma=1.5).fit(X_train)
scores = model.score_samples(X_test)             # numpy array of Born-rule scores
auc    = roc_auc(model.score_samples(X_normal),
                 model.score_samples(X_anomaly))

model.partial_fit(X_drift, alpha=0.01)           # streaming EMA update
model.calibrate(X_train, contamination=0.05)
preds = model.predict(X_test)                    # +1 inlier, -1 outlier
```

## Benchmark results

Synthetic 4-D data, 400 training points, 300+300 test, `D = 256`, σ = 1.5,
seed = 42. Mahalanobis is included as a strong linear baseline.

| Scenario                              | DMKDE AUC | Mahalanobis AUC | Verdict           |
|---------------------------------------|-----------|-----------------|-------------------|
| Gaussian normal                       | **1.000** | **1.000**       | tied              |
| Bimodal manifold (anomalies in gap)   | **1.000** | 0.439           | DMKDE +0.56 AUC   |
| Ring manifold (anomalies in hub)      | **0.836** | 0.000           | DMKDE +0.84 AUC   |
| Streaming (after 200 EMA updates)     | **1.000** | n/a             | converges         |

Mahalanobis is mathematically optimal on Gaussian data and ties there.
On any data where the "normal" class has higher-order structure (multi-modal,
manifold, non-convex) Mahalanobis collapses — on the ring scenario it scores
exactly 0.000 AUC because anomalies sit at the empirical mean. DMKDE captures
the structure because RFF + density-matrix scoring is a kernelized density
estimator, not a Gaussian-fit.

## Roadmap

- [x] pybind11 Python bindings with sklearn-style `fit` / `score_samples`
- [x] PyOD plugin (`dmkde.pyod.DMKDEDetector`)
- [x] Qiskit backend (`dmkde.qiskit_backend.QiskitDMKDE`) — reproduces
      classical scores to floating-point precision via amplitude
      encoding + statevector estimator
- [x] Benchmarks vs sklearn baselines on Kaggle Credit Card Fraud +
      KDDCup'99 — see [`BENCHMARKS.md`](BENCHMARKS.md)
- [x] cibuildwheel CI building wheels for Linux + macOS + Windows ×
      CPython 3.9–3.13
- [ ] PyPI release (sdist + multi-platform wheels)
- [ ] Qiskit backend hardware execution (transpile + EstimatorV2 +
      shot budgeting, run on IBM QPU)
- [ ] Latent variant (LADDM autoencoder pre-stage, arXiv:2408.07623)
- [ ] NSL-KDD + IEEE-CIS Fraud + CICIDS-2018 benchmarks

## Benchmarks

See [`BENCHMARKS.md`](BENCHMARKS.md) for full results across five
detectors and five scenarios. Headline:

- **KDDCup'99 intrusion** — DMKDE wins (ROC 0.9965, PR 0.9807)
- **Ring manifold** — DMKDE (0.96) and LOF (0.99) are the only methods
  that don't collapse to 0.000 AUC
- **Credit Card Fraud** — DMKDE 0.95 ROC, behind Mahalanobis (0.96)
  because V1–V28 are PCA-pretreated to be approximately Gaussian

## Qiskit backend

The score `⟨φ(x)|ρ|φ(x)⟩` is the expectation value of ρ in the state
|φ(x)⟩ — an actual quantum observable on a quantum state. The optional
Qiskit backend amplitude-encodes φ(x) on `⌈log₂ D⌉` qubits and evaluates
the same expectation via `qiskit.quantum_info.Statevector`:

```python
from dmkde import DMKDE
from dmkde.qiskit_backend import QiskitDMKDE

model  = DMKDE(feature_dim=16, sigma=1.5).fit(X_train)  # D = 2^4
qmodel = QiskitDMKDE(model)
q_score = qmodel.score(x_test)                          # uses StatePreparation + statevector
c_score = model.score_samples(x_test[None, :])[0]
assert abs(q_score - c_score) < 1e-10                   # matches to FP precision

print(qmodel.to_circuit(x_test).draw())                 # the 4-qubit circuit
```

`pip install qiskit` to enable.

## PyOD plugin

```python
from dmkde.pyod import DMKDEDetector
det = DMKDEDetector(feature_dim=256, sigma=1.5, contamination=0.05).fit(X_train)
labels = det.predict(X_test)        # 0 = inlier, 1 = outlier (PyOD convention)
scores = det.decision_function(X_test)
```

`pip install pyod` to enable.

## Santander X Quantum AI Leap submission

[`SUBMISSION.md`](SUBMISSION.md) contains the full submission narrative
for the Santander X Global Challenge: Quantum AI Leap (June 30 2026
deadline).

## License

MIT — see [`LICENSE`](LICENSE). Provided AS IS with no warranty.
