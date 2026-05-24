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

```sh
make           # builds benchmark + tests
make test      # runs the unit tests
make bench     # runs the reproducible benchmark
```

Requires `g++` or `clang++` with C++17. No external dependencies.

## Use

```cpp
#include "dmkde.hpp"

// 4 input dims, 256-D RFF embedding, Gaussian kernel bandwidth σ = 1.5.
dmkde::DMKDE model(/*input_dim=*/4, /*feature_dim=*/256, /*sigma=*/1.5);

std::vector<std::vector<double>> train = /* normal samples */;
model.fit(train);

double s   = model.score(new_point.data());       // higher = more "normal"
model.update(drift_point.data(), /*alpha=*/0.01); // streaming EMA update
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

- pybind11 Python bindings with sklearn-style `fit` / `score_samples`
- PyOD plugin registration
- Qiskit / PennyLane backend that runs the Born measurement on quantum
  hardware (the embedding maps 1:1 to a parameterised circuit)
- Latent variant (LADDM autoencoder pre-stage, arXiv:2408.07623)
- Tier-1 fraud-detection dataset benchmarks (Kaggle credit-card,
  KDDCup'99, NSL-KDD)

## License

MIT — see [`LICENSE`](LICENSE). Provided AS IS with no warranty.
