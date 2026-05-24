# Santander X Global Challenge — Quantum AI Leap Submission

## Project: `libdmkde`

Production-grade implementation of **density-matrix kernel density
estimation** — a quantum-derived anomaly-detection algorithm — packaged
as a single-header C++17 library with Python bindings, sklearn/PyOD
compatibility, and an optional Qiskit backend for execution on real IBM
quantum hardware.

**Repository:** https://github.com/EmpowerTours/libdmkde
**License:** MIT
**Application area:** Pillar 2 — Solutions that combine quantum
computing and AI to solve real-world problems

---

## The problem

Anomaly detection drives some of the highest-value workflows in banking:
fraud detection, AML signal generation, intrusion detection on internal
systems, and detection of compromised devices on the IoT perimeter.
Production stacks today rely on Isolation Forest, One-Class SVM, and
autoencoder ensembles — all of which collapse on data with non-Gaussian,
multi-modal, or manifold structure. The failure mode is well documented
and not avoidable through hyperparameter tuning.

## The algorithm

In 2024, Useche, González, et al. (*Phys. Rev. A* **109**, 032418)
introduced a quantum-mechanically motivated detector that maps each
input through a Random-Fourier-Features embedding to a quantum-state
vector |φ(x)⟩, builds the empirical density operator

    ρ = (1/N) Σᵢ |φ(xᵢ)⟩⟨φ(xᵢ)|

and scores test points by the Born-rule probability

    s(x) = ⟨φ(x)| ρ |φ(x)⟩.

The score is the literal quantum-mechanical probability of measuring x
in the training subspace. Low s ⇒ anomalous.

Three published papers extend this base — AD-DMKDE (deep variant,
arXiv:2210.14796), LADDM (latent variant, arXiv:2408.07623), and INQMAD
(streaming O(1) update, arXiv:2210.05061). **All four exist only as
research Jupyter notebooks** — single-digit star counts, no PyPI
packages, no sklearn/PyOD API, no library a bank can deploy.

## What `libdmkde` adds

We take the published algorithm and ship the engineering that has been
missing. Specifically:

### Engineering
- **Single-header C++17 core** — no external dependencies, builds with
  `g++` or `clang++` in under a second; drop `dmkde.hpp` into any
  project.
- **pybind11 bindings** that expose the C++ core to Python with numpy
  zero-copy interop, sklearn-style `fit`/`score_samples`/
  `decision_function`/`predict` API, and a `partial_fit` for the INQMAD
  rank-1 EMA streaming update.
- **PyOD plugin** (`dmkde.pyod.DMKDEDetector`) registers DMKDE as a
  PyOD-compatible detector — drops into existing PyOD ensembles.
- **`pip install dmkde`** — cibuildwheel CI builds and tests wheels for
  Linux + macOS + Windows × CPython 3.9–3.13 on every push to main.
- **Reproducible benchmarks** against scikit-learn baselines and on two
  public real datasets (Kaggle Credit Card Fraud and KDDCup'99) —
  numbers in [BENCHMARKS.md](BENCHMARKS.md).

### Quantum integration
- **Optional Qiskit backend** (`dmkde.qiskit_backend.QiskitDMKDE`)
  amplitude-encodes φ(x) onto ⌈log₂ D⌉ qubits via `StatePreparation`,
  constructs ρ as a `qiskit.quantum_info.Operator` observable, and
  evaluates ⟨φ|ρ|φ⟩ via Qiskit's statevector estimator. Verified to
  reproduce classical scores to floating-point precision (max diff
  ≈ 3 × 10⁻¹⁷). Swap the simulator for an IBM backend for hardware
  execution.

This makes `libdmkde` the **only known production-grade implementation
of the Useche 2024 algorithm**, and the first to ship a clean
classical-or-quantum hybrid execution model for this family of
detectors.

## Validated results

### KDDCup'99 network intrusion (3.36% anomaly rate)

| Detector | ROC-AUC | PR-AUC |
|---|---:|---:|
| **DMKDE** | **0.9965** | **0.9807** |
| Mahalanobis | 0.9960 | 0.9791 |
| OneClassSVM | 0.9951 | 0.9799 |
| LOF | 0.9805 | 0.8627 |
| Isolation Forest | 0.9584 | 0.8828 |

DMKDE wins both metrics on real intrusion-detection data.

### Synthetic ring (manifold-anomaly diagnostic)

Normal data on a 4-D hypersphere of radius 3; anomalies cluster at the
centre — the empirical mean. Mahalanobis, Isolation Forest, and
OneClassSVM all score **exactly 0.000 AUC** because their "most normal"
prediction lives at the mean, which is exactly where the anomalies are.
Only DMKDE (0.962) and LOF (0.988) survive.

### Kaggle Credit Card Fraud (0.17% anomaly rate)

| Detector | ROC-AUC | PR-AUC |
|---|---:|---:|
| Mahalanobis | 0.9616 | 0.8720 |
| DMKDE | 0.9546 | 0.7391 |
| LOF | 0.9512 | 0.8443 |
| OneClassSVM | 0.9509 | 0.8217 |
| Isolation Forest | 0.9490 | 0.6599 |

Mahalanobis wins here because V1–V28 are PCA-orthogonalised — the
"normal" class is approximately Gaussian post-PCA, which is exactly the
regime Mahalanobis was designed for. DMKDE places second, ahead of all
non-linear baselines. We report this honestly rather than hiding it.

## Why this fits the challenge

**Short-to-medium-term impact:** the C++ library runs today on a single
CPU core with no quantum hardware. The Qiskit backend lets a bank's
risk team validate that their pipeline gives identical results when
offloaded to an IBM QPU, providing an upgrade path that does not
require waiting for fault-tolerant systems.

**Genuine quantum content:** the score is the Born-rule probability —
not a classical analogy. The embedding maps 1:1 to a parameterised
circuit; ρ is a literal density operator with trace 1 and verifiable
positive-semidefinite structure. IBM/Bluzec judges can evaluate the
quantum semantics directly.

**Production-ready:** sklearn/PyOD-compatible, pip-installable with
prebuilt wheels, benchmark suite, reproducible numbers, MIT license.

## Roadmap if funded

1. **Tier-1 fraud benchmark expansion** — IEEE-CIS Fraud, ULB Credit
   Card, NSL-KDD, CICIDS-2018. Publish numbers + reproducibility
   scripts.
2. **Qiskit backend hardware execution** — transpile via Qiskit
   Runtime, integrate `EstimatorV2` with shot budgeting, run on ibm_kingston
   (or whatever the highest-priority backend is at the time of the
   pilot), validate vs simulator.
3. **PyOD upstream integration** — submit the detector to the official
   PyOD model catalogue for inclusion in the next release.
4. **Latent variant (LADDM)** — autoencoder pre-stage from
   arXiv:2408.07623 for high-dimensional images / log streams.
5. **Streaming dashboards** — Kafka + Apache Flink connector for live
   transaction streams; ρ updates per-event in O(D²) without retraining.

## Team

EmpowerTours, single-founder development. Contact via GitHub issues at
https://github.com/EmpowerTours/libdmkde.

## Citation guidance

This library is an **engineering artefact**, not a research paper.
Anyone using it in academic work should cite the original authors:

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
```

See [`CITATION.cff`](CITATION.cff) for the full machine-readable list.
