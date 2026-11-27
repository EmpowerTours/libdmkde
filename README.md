# libdmkde

A production-grade C++17 implementation of **Density-Matrix Kernel Density
Estimation** for anomaly detection, with a streaming rank-1 update
(O(D²) time and O(1) extra memory per sample).

The algorithm is **not new** — it was introduced by Useche, Bustos-Brinez,
Gallego-Mejia and González in *Phys. Rev. A* (2024). What this repository provides is the engineering
that has been missing from the published research code: a single-header C++
library, a deterministic benchmark harness, reproducible numbers against a
classical baseline, and an MIT-licensed build that a production team can
adopt without untangling Jupyter notebooks.

## Citation — please cite the original authors

If you use this library in academic work, **cite the papers, not this repo:**

```bibtex
@article{useche2024quantum,
  author  = {Useche, Diego H. and Bustos-Brinez, Oscar A. and
             Gallego-Mejia, Joseph A. and Gonz{\'a}lez, Fabio A.},
  title   = {Quantum density estimation with density matrices:
             Application to quantum anomaly detection},
  journal = {Physical Review A},
  volume  = {109},
  pages   = {032418},
  year    = {2024},
  doi     = {10.1103/PhysRevA.109.032418}
}

@misc{gallegomejia2022inqmad,
  author = {Gallego-Mejia, Joseph and Bustos-Brinez, Oscar and
            Gonz{\'a}lez, Fabio A.},
  title  = {InQMAD: Incremental Quantum Measurement Anomaly Detection},
  year   = {2022},
  note   = {arXiv:2210.05061}
}

@misc{bustosbrinez2022addmkde,
  author = {Bustos-Brinez, Oscar and Gallego-Mejia, Joseph and
            Gonz{\'a}lez, Fabio A.},
  title  = {AD-DMKDE: Anomaly Detection through Density Matrices
            and Fourier Features},
  year   = {2022},
  note   = {arXiv:2210.14796}
}
```

The density-matrix estimator built from random Fourier features goes back
to earlier work by the same group, which the papers above build on:

```bibtex
@article{gonzalez2022learning,
  author  = {Gonz{\'a}lez, Fabio A. and Gallego, Alejandro and
             Toledo-Cort{\'e}s, Santiago and Vargas-Calder{\'o}n, Vladimir},
  title   = {Learning with Density Matrices and Random Features},
  journal = {Quantum Machine Intelligence},
  volume  = {4},
  pages   = {23},
  year    = {2022},
  doi     = {10.1007/s42484-022-00079-9},
  note    = {arXiv:2102.04394}
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

ranks `x` by its density under the training distribution. Low `s` ⇒
anomalous. Two properties worth knowing:

- **Effective bandwidth is σ/√2.** `s(x) = (1/N) Σᵢ ⟨φ(x), φ(xᵢ)⟩²` is a
  mean of the *squared* kernel, `k(x, xᵢ)² = exp(-‖x - xᵢ‖² / σ²)`, i.e. a
  Gaussian KDE with bandwidth `σ/√2`, not `σ`.
- **The score is an unnormalized density estimate.** It lies in [0, 1] and
  does not integrate to 1 (the paper's `1/M_γ` normalization is not
  applied). Rankings, ROC-AUC and thresholds are unaffected.

The RFF frequencies are a fixed seeded draw (not learned). Since 0.2.0 the
draw is portable: a seed gives the same frequencies on Linux, macOS and
Windows (0.1.x used `std::normal_distribution`, which differs between
standard libraries, so a seed now builds a different model than in 0.1.x;
saved 0.1.x models load unchanged because they store the weights). The streaming
variant updates `ρ` via a rank-1 EMA:

    ρ ← (1 - α) ρ + α · φ(x) φ(x)ᵀ

with `O(D²)` time and `O(1)` extra memory per sample. Updates are
unconditional (InQMAD's threshold gating is not implemented).

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

double s = model.score(new_point);                // size-checked; higher = more "normal"
model.update(drift_point, /*alpha=*/0.01);         // size-checked streaming EMA update
```

Overloads taking `std::vector<double>` or a pointer plus a length check the
width and reject NaN / inf. The pointer-only forms (`score(const double*)`,
`update(const double*, alpha)`, `transform(const double*, double*)`) are the
unchecked fast path: the caller guarantees `input_dim()` doubles. Invalid
parameters (`sigma` not finite and > 0, `input_dim <= 0`, odd `feature_dim`,
`feature_dim > dmkde::kMaxFeatureDim` = 2¹⁴, or more than
`dmkde::kMaxWeightCount` = 2²⁸ random-feature weights `(feature_dim/2)·input_dim`,
e.g. `input_dim > 2²¹` at `feature_dim = 256`, without `allow_large`) throw
`std::invalid_argument`; using a model before `fit` throws
`std::logic_error`. The checked `Mahalanobis` score paths throw
`std::invalid_argument` when a finite query's distance exceeds `DBL_MAX`
(~1.8e308, e.g. a query near ±1e308 along a direction of small variance);
the unchecked `score(const double*)` returns `-inf` there.

### Python (sklearn-style)

```python
import numpy as np
from dmkde import DMKDE, Mahalanobis, roc_auc

model = DMKDE(feature_dim=256, sigma=1.5, contamination=0.05).fit(X_train)
scores = model.score_samples(X_test)             # numpy array of Born-rule scores
auc    = roc_auc(model.score_samples(X_normal),
                 model.score_samples(X_anomaly))

preds = model.predict(X_test)                    # +1 inlier, -1 outlier
df    = model.decision_function(X_test)          # score - offset_; < 0 => outlier

model.partial_fit(X_drift, alpha=0.01)           # streaming EMA update (alpha by keyword)
model.calibrate(X_recent)                        # optional: re-set offset_ on held-out data
```

`DMKDE` is a scikit-learn estimator (`get_params` / `set_params`, `clone`,
pipelines, pickling). It passes `sklearn.utils.estimator_checks.check_estimator`
except `check_outliers_train` and `check_outliers_fit_predict`, which require
`predict` on the *training* set to flag exactly `contamination` of it; DMKDE
is a novelty detector whose threshold is calibrated for new data (see below),
the case scikit-learn itself exempts for `LocalOutlierFactor(novelty=True)`.
`fit` sets `offset_` from the **leave-one-out** training scores (below), and
`decision_function` follows the sklearn convention (negative ⇒ outlier,
`predict == sign`). Inputs must be 2-D, finite, real and have
`n_features_in_` columns. Error types (they follow scikit-learn's
`validate_data`):

- `ValueError`: wrong rank or width, NaN / inf, empty input, strings that are
  not numbers, complex data, and values too large for a double (including
  Python ints such as `10**400`, which 0.1.x let through as `OverflowError`);
  invalid parameters (`sigma`, `feature_dim`, `contamination`, `alpha`,
  `threshold`, `random_state`).
- `TypeError`: sparse matrices (dense data is required), objects NumPy
  cannot read as numbers at all (e.g. a dict), DataFrame column names that
  mix strings and non-strings, and a value passed positionally where
  `partial_fit` expects `y` (pass `alpha=` by keyword).
- `NotFittedError` (a subclass of `ValueError` and `AttributeError`) when
  scoring before `fit`.

The low-level `dmkde._core` functions raise `ValueError` for every invalid
array (including non-numeric ones) and `RuntimeError` before `fit`.
`Mahalanobis.score_samples` raises `ValueError` when a finite query's
distance exceeds `DBL_MAX` (~1.8e308) rather than returning `-inf`.

`Mahalanobis` has no threshold: its `decision_function` is `score_samples`
unchanged (the negative distance, always ≤ 0, no `offset_`, no `predict`),
unlike `DMKDE`'s `score_samples − offset_`. Its ridge is relative per
feature, so non-constant features are unit-invariant; a feature that is
constant in training uses an absolute scale (inverse variance `1e6`, so a
deviation δ alone is distance `1000·|δ|`, independent of the other columns
but not of that feature's own units). See
[`CHANGELOG.md`](https://github.com/EmpowerTours/libdmkde/blob/main/CHANGELOG.md) for the 0.1.1 behaviour changes.

### Threads

Scoring is safe from many threads at once, also while `partial_fit` updates
the same model (a per-model reader/writer lock; the GIL is released). Do not
call `fit` or `calibrate` on an estimator other threads are scoring with:
fit a new estimator and swap the reference instead
(`model = DMKDE(...).fit(X_new)`).

### Thresholds, σ and the noise floor

- **Threshold.** Each training row's in-sample score contains its own term,
  exactly `1/n_samples` (every RFF embedding has unit norm), so in-sample
  scores sit above the scores of fresh data. `fit` therefore uses the exact
  leave-one-out scores `(n·s − 1)/(n − 1)`, and sets `offset_` to the k-th
  smallest of them with `k = floor((N + 1)·contamination)`. By the
  conformal rank argument this flags on average about `contamination` or
  less of fresh normal data (`k/(N + 1)`, e.g. 0.04995 at N = 1000,
  c = 0.05). The bound is exact for held-out scores (`calibrate`) and
  approximate, to O(1/N), for leave-one-out scores (`fit`); with many tied
  scores (discrete data) the rate can fall well below `contamination`. `np.quantile` would over-flag at small N (0.37-0.44 at N = 2, 0.07-0.13
  at N = 20). **Caveat for small N:** below `N = 1/contamination − 1` (19 at
  0.05) no threshold can meet the target, so nothing is flagged and `fit`
  warns; while `N·contamination < 5` the rate of a single fit is noisy (only
  its average is bounded) and `fit` warns too. `calibrate` applies the same
  rule to whatever data you give it, so give it held-out data, not the
  training set.
- **Standardize the features, and let σ grow with the dimension.** The
  default `sigma=1.0` suits only a few standardized features. A rule of thumb
  is `sigma ≈ sqrt(d)` for `d` standardized features, or the median pairwise
  distance of the standardized training data divided by `sqrt(2)`.
- **Noise floor ≈ 1/D.** With random Fourier features, two points that are
  far apart relative to σ still have a squared feature overlap of about
  `1/D` (`D = feature_dim`) on average. When σ is too small every score sits
  at that noise floor and the ranking is close to random (at the default σ
  with 20+ standardized features, AUC ≈ 0.5). `fit` emits a `UserWarning`
  when the median leave-one-out training score is below `3/D`. Measured at
  N = 500, median·D of 1.1-1.5 lost 0.07-0.18 AUC to the exact density; with
  very few rows the warning is not sufficient (at N = 10, fits with median·D
  3-6 and no warning still lost up to 0.09 AUC).
- **Duplicate training rows.** The leave-one-out score of a row that occurs
  k times still contains its k - 1 copies, each contributing a kernel value
  of 1, while a fresh point has no exact copy. With many duplicates and a
  small σ the leave-one-out scores (and so `offset_`) sit far above the
  scores of fresh data and `predict` over-flags: measured, 150 distinct
  6-D points each repeated 3 times at σ = 0.8 flagged 76% of fresh normal
  data at `contamination=0.05`. Either deduplicate first
  (`np.unique(X, axis=0)`: 5.5% flagged) or `calibrate` on held-out data
  (5.1%). A larger σ (≈ `sqrt(d)`) also shrinks the effect (6.7%).
- **σ too large.** When σ is large compared with the spread of the data,
  every score is close to 1 and DMKDE degrades into a distance-to-the-mean
  detector: on two separated clusters, anomalies between them score as the
  most normal points (measured AUC 0.0003). `fit` warns when 1 − median
  leave-one-out score < 0.02, and more strongly when it is below 1e-9,
  where scores start to round to exactly 1 and tie.
- **Streaming forgets fast.** `partial_fit` weights the existing model by
  `(1 − α)` per row, so its memory is about `1/α` rows: one 1000-row batch at
  `alpha=0.01` leaves the original fit a weight of `0.99¹⁰⁰⁰ ≈ 4e-5`.

## Benchmark results

Synthetic 4-D data (`python python/benchmark.py`): 400 training points,
300+300 test, `D = 256`, σ = 1.5, one split with seed 42. Mahalanobis is
included as a strong linear baseline; LOF is the strongest of the other
baselines in [`BENCHMARKS.md`](https://github.com/EmpowerTours/libdmkde/blob/main/BENCHMARKS.md).

| Scenario                              | DMKDE AUC | Mahalanobis AUC | LOF AUC |
|---------------------------------------|-----------|-----------------|---------|
| Gaussian normal                       | 1.000     | 1.000           | 1.000   |
| Bimodal manifold (anomalies in gap)   | 1.000     | 0.436           | 1.000   |
| Ring manifold (anomalies in hub)      | 0.962     | 0.000           | 0.988   |
| Streaming (`python/example.py`, after 300 EMA updates) | 1.000 | n/a | n/a |

Mahalanobis is mathematically optimal on Gaussian data and ties there.
Where the "normal" class has higher-order structure (multi-modal, manifold,
non-convex) Mahalanobis collapses; on the ring it scores exactly 0.000 AUC
because the anomalies sit at the empirical mean. DMKDE captures the
structure because RFF + density-matrix scoring is a kernelized density
estimator, not a Gaussian fit; LOF does too, and beats DMKDE on the ring on
every seed from 0 to 9. These are single-split numbers with σ chosen by
hand; see [`BENCHMARKS.md`](https://github.com/EmpowerTours/libdmkde/blob/main/BENCHMARKS.md) for the seed variation.
(`make bench` runs a C++ demo with a different data generator, so its
numbers differ.)

## Roadmap

- [x] pybind11 Python bindings with sklearn-style `fit` / `score_samples`
- [x] PyOD plugin (`dmkde.pyod.DMKDEDetector`)
- [x] Qiskit cross-check (`dmkde.qiskit_backend.QiskitDMKDE`) — reproduces
      classical scores to floating-point precision by exact statevector
      simulation (`Statevector.expectation_value`)
- [x] Benchmarks vs sklearn baselines on Kaggle Credit Card Fraud +
      KDDCup'99 — see [`BENCHMARKS.md`](https://github.com/EmpowerTours/libdmkde/blob/main/BENCHMARKS.md)
- [x] cibuildwheel CI building wheels for Linux + macOS + Windows ×
      CPython 3.9–3.13
- [x] PyPI release (`pip install dmkde`; sdist + multi-platform wheels)
- [ ] Qiskit backend hardware execution (transpile + EstimatorV2 +
      shot budgeting, run on IBM QPU)
- [ ] Latent variant (LADDM autoencoder pre-stage, arXiv:2408.07623)
- [ ] NSL-KDD + IEEE-CIS Fraud + CICIDS-2018 benchmarks

## Benchmarks

See [`BENCHMARKS.md`](https://github.com/EmpowerTours/libdmkde/blob/main/BENCHMARKS.md) for full results across five
detectors and six scenarios. All published numbers are one train/test split
with seed 42, and DMKDE's σ was chosen by hand on that split (no held-out
selection), so read them as indicative. Headline:

- **KDDCup'99 intrusion** (numeric columns only; the categorical ones are
  dropped) — on seed 42 DMKDE is highest (ROC 0.9965, PR 0.9807), but
  within 0.001 of Mahalanobis and OneClassSVM; on seeds 1 and 2
  OneClassSVM is best. Treat the top three as tied.
- **Ring manifold** — DMKDE (0.96) and LOF (0.99) are the only methods
  that don't collapse to 0.000 AUC; LOF is higher on every seed 0-9
- **Credit Card Fraud** — DMKDE 0.95 ROC, behind Mahalanobis (0.96)
  because V1–V28 are PCA-pretreated to be approximately Gaussian, but
  above LOF and OneClassSVM; by PR-AUC DMKDE is 4th of 5 (0.74, 0.10
  behind LOF)

## Qiskit backend

The score `⟨φ(x)|ρ|φ(x)⟩` is the expectation value of ρ in the state
|φ(x)⟩ — an actual quantum observable on a quantum state. The optional
Qiskit backend builds the `⌈log₂ D⌉`-qubit state |φ(x)⟩ and evaluates the
same expectation exactly with `qiskit.quantum_info.Statevector` and a dense
`Operator(ρ)`. This is classical statevector simulation for cross-checking;
it does not use the Estimator primitives and has no hardware path yet:

```python
from dmkde import DMKDE
from dmkde.qiskit_backend import QiskitDMKDE

model  = DMKDE(feature_dim=16, sigma=1.5).fit(X_train)  # D = 2^4
qmodel = QiskitDMKDE(model)
q_score = qmodel.score(x_test)                          # exact statevector expectation
c_score = model.score_samples(x_test[None, :])[0]
assert abs(q_score - c_score) < 1e-10                   # matches to FP precision

print(qmodel.to_circuit(x_test).draw())                 # StatePreparation circuit (not executed)
```

`pip install qiskit` to enable.

## PyOD plugin

```python
from dmkde.pyod import DMKDEDetector
det = DMKDEDetector(feature_dim=256, sigma=1.5, contamination=0.05).fit(X_train)
labels = det.predict(X_test)        # 0 = inlier, 1 = outlier (PyOD convention)
scores = det.decision_function(X_test)
```

`pip install pyod` to enable. `DMKDEDetector`'s `contamination` defaults to
0.1, PyOD's own default for every detector, so it is interchangeable with
the other PyOD models; the native `dmkde.DMKDE` defaults to 0.05. Pass
`contamination` explicitly to get the same threshold from both.

### Saving models

`pickle` works for both estimators; the low-level `_core.DMKDE.get_state()`
/ `from_state()` export and import a plain dict. Saved states contain the
random-feature weights `W`, so they load identically on every platform and
from 0.1.x. `from_state` validates every field and, for a fitted state,
accepts ρ only within these tolerances: `|trace(ρ) − 1| <= 1e-6`, symmetry
`|ρ_ij − ρ_ji| <= 1e-12·(1 + |ρ_ij|)`, and positive semidefinite with
smallest eigenvalue `> −1e-9·trace` (proved by Cholesky for
`feature_dim <= 2048`; above that a randomised probe that can miss a tiny
negative eigenvalue, so only load large states from trusted sources).
Unpickling also checks that `offset_` is a finite float in [0, 1] and that
`n_features_in_` matches the model.

## License

MIT — see [`LICENSE`](https://github.com/EmpowerTours/libdmkde/blob/main/LICENSE). Provided AS IS with no warranty.
