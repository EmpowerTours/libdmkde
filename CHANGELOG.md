# Changelog

## 0.1.1 — safety release

Fixes the flaws found by two independent audits of 0.1.0 and five further
review rounds. For valid input `DMKDE.score_samples` /
`_core.DMKDE.score_batch` return bit-identical values to 0.1.0 (same seed ⇒
same RFF draw, same accumulation order), so every DMKDE ROC-AUC / PR-AUC
in BENCHMARKS.md reproduces unchanged. What changes by design is the threshold:
`offset_`, and with it `predict` and `decision_function` (see below).

### Fixes

**Memory safety and shapes**
- Every binding entry point that takes a row or matrix now checks rank and
  width against the model before any pointer reaches C++: `DMKDE.score`,
  `score_batch`, `transform`, `update`, `fit`; `Mahalanobis.fit`, `score`,
  `score_batch`; `roc_auc`. 0.1.0 read past the caller's buffer for narrow
  input (heap out-of-bounds read) and dereferenced NULL for `(N, 0)` input
  (segfault, also in the published PyPI wheel).
- `(N, 0)` arrays and 0-feature models are rejected.
- `_core.roc_auc` converts its inputs with `c_style | forcecast`; 0.1.0 read
  a strided view such as `x[::2]` as raw contiguous memory.
- C++ API: new size-checked overloads (`score(const std::vector<double>&)`,
  `score(const double*, size_t)`, `score_batch(...)`, `update(vector, α)`,
  `update(const double*, size_t, α)`, `transform(vector)`,
  `fit(const double*, rows, cols)`, Mahalanobis equivalents). The
  pointer-only forms remain as the documented, unchecked fast path.
- Python layer validates input with scikit-learn's `validate_data` (or an
  equivalent fallback), enforces `n_features_in_` and raises sklearn-style
  errors.

**Non-finite input**
- NaN / inf are rejected in `fit`, `partial_fit` / `update`, Mahalanobis
  `fit` / `score`, scoring, and `roc_auc` (`ValueError`). In 0.1.0 one NaN
  training value turned every score into NaN and `predict` flagged every
  point; a NaN Mahalanobis query scored `-0.0`, the most "normal" value.
- A rejected `update` leaves ρ unchanged; `update` with `alpha = NaN` is
  rejected (0.1.0 accepted it and poisoned ρ).
- The raw C++ `Mahalanobis::score` returns NaN (never `-0.0`) for a NaN row.
- Finite but huge input (e.g. `1.7e308`) whose RFF embedding overflows is
  rejected by `score_samples` / `decision_function` / `predict` /
  `calibrate` and by `_core` `score` / `score_batch` (`ValueError`, "RFF
  embedding of x is not finite"), exactly as `fit` / `update` already did.
  Before, it passed the finiteness check, scored NaN and, through
  `calibrate`, set `offset_ = NaN` so `predict` flagged every point.
- Mahalanobis: when the quadratic form overflows for a finite query (e.g.
  `[1e308, -1e308, 0]`, which gave NaN), it is recomputed on the rescaled
  difference, giving a very large finite distance. If `x - mean` itself
  overflows the checked paths raise `ValueError`. Scores never come back NaN.
- `partial_fit` is atomic: the whole batch (finite input and finite RFF
  embeddings) is validated before any row is applied, so a rejected batch
  leaves ρ and `n_streamed` unchanged. It uses the new
  `_core.DMKDE.update_batch` / C++ `DMKDE::update_batch`, which gives the same
  ρ as per-row `update` calls.
- A refit that fails (in validation or in the C++ core) leaves the previous
  fitted state intact: `n_features_in_`, `_impl` and `offset_` are replaced
  only after the new fit succeeded (`DMKDE` and `Mahalanobis`).

**Parameters**
- `sigma` must be finite and > 0; `input_dim > 0`; `feature_dim` positive,
  even and at most `MAX_FEATURE_DIM = 2**14` (ρ needs D²·8 bytes = 2 GiB at
  the cap) unless `allow_large_feature_dim=True` (Python) /
  `allow_large = true` (C++). Parameters are checked before ρ is allocated.
- `Mahalanobis` needs at least 2 training rows (0.1.0 divided by N-1 = 0 and
  scored every point, however far, as perfectly normal). Training data with
  zero variance in every feature is rejected.
- `update` / `score` before `fit` raise (`RuntimeError` from `_core`,
  `NotFittedError` from the estimator).
- `contamination` (fit and `calibrate`) must be in (0, 0.5].

**scikit-learn contract**
- `DMKDE` subclasses `BaseEstimator` and `OutlierMixin` when scikit-learn is
  importable (a minimal fallback otherwise; scikit-learn stays optional):
  `get_params` / `set_params`, `clone`, pipelines and grid search work.
  `Mahalanobis` passes `sklearn.utils.estimator_checks.check_estimator`;
  `DMKDE` passes every check except `check_outliers_train` and
  `check_outliers_fit_predict` (expected for a novelty detector, see
  "Behaviour changes"). Sparse input raises `TypeError` ("dense data is
  required"), as in scikit-learn; every other invalid input raises
  `ValueError`.
- `random_state` accepts None, an int, a `RandomState` or a `Generator`. An
  int is used as the C++ seed directly, so `random_state=42` gives the same
  model as 0.1.0.
- Pickling: `DMKDE` and `Mahalanobis` pickle via `__getstate__` /
  `__setstate__`. The state stores the RFF weights `W`, ρ and the counters
  (not just the seed), so a pickle loads identically on any platform. New
  `_core.DMKDE.get_state()` / `from_state()` (and Mahalanobis equivalents)
  validate every field on import: `W` and ρ finite, ρ symmetric with trace 1
  and positive semidefinite (smallest eigenvalue > -1e-9·trace; see
  "Loading cost" below for how); Mahalanobis `mean` finite and `inv_cov`
  finite, symmetric and positive definite (Cholesky). Malformed states (not a
  dict, missing keys, wrong types such as a float `format` or a string
  `fitted`, negative `seed`, out-of-range dimensions or counters, non-numeric
  arrays) raise `ValueError`, not a `RuntimeError` cast error.
- `fit` resets any threshold set by `calibrate`.
- `_core.DMKDE.fit` resets `n_streamed`.
- `DMKDE` without scikit-learn (the fallback base class) now records
  `feature_names_in_` for a DataFrame and raises `ValueError` when a later
  call passes the columns reordered, renamed or missing (and warns when names
  appear or disappear), as scikit-learn does. Before, reordered columns were
  silently scored as if in the original order.
- PyOD adapter: `DMKDEDetector` is one cached module-level class, so
  `isinstance` and pickling work; `predict_proba` works; an unfitted
  `decision_function` raises `NotFittedError`; input width is checked.
- Qiskit backend: `encode` checks width and finiteness; ρ is re-read from the
  model whenever it changed (it was a stale snapshot taken at construction).
  The power-of-two check and the qubit count use the fitted D
  (`model._impl.feature_dim`), not the `feature_dim` parameter, which
  `set_params` can change after fit; a model refit to a different D after
  wrapping raises `ValueError` instead of being mis-simulated.

**GIL and locking**
- `fit`, `score_batch`, `update`, `from_state`, Mahalanobis `fit` /
  `score_batch` / `from_state` and `roc_auc` release the GIL (Mahalanobis
  `from_state`, whose O(d³) check held the GIL in the 0.1.1 release
  candidate, too). Each model has a phase-fair reader/writer lock (mutex +
  condition variable): scoring runs in parallel across threads while `fit` /
  `update` are exclusive, writers are served in FIFO order, a waiting writer
  blocks new readers, and a releasing writer admits every queued reader
  before the next writer. Neither side can starve the other: continuous
  scoring no longer delays `update` / `partial_fit` for seconds
  (`std::shared_mutex` in libstdc++ prefers readers), and back-to-back
  writers no longer starve `score` (the release candidate's
  writer-preferring lock made one `score` call wait 8-10 s behind 2-3 busy
  writers at D = 1024; now the longest wait is 35-63 ms, about one update).
  This lock covers one fitted model (`_core`): concurrent scoring with
  `partial_fit` / `update_batch` on the same estimator is safe. Re-running
  `fit` (or `calibrate`) on an estimator that other threads are scoring with
  is **not** supported, as for scikit-learn estimators in general: `fit`
  installs a new model and a new `offset_` separately, so a concurrent
  `predict` can pair one with the other. To retrain a live model, fit a new
  estimator and swap the reference (`model = DMKDE(...).fit(X_new)`), which is
  a single atomic assignment.

**Docs and metadata**
- The earlier density-matrix / random-Fourier-feature density estimation
  work is now credited (README, CITATION.cff): González, Gallego,
  Toledo-Cortés, Vargas-Calderón, *Learning with Density Matrices and Random
  Features*, Quantum Mach. Intell. 4, 23 (2022), arXiv:2102.04394.
- Citations corrected everywhere: arXiv:2201.10006 is by Useche,
  Bustos-Brinez, Gallego-Mejia and González; InQMAD (arXiv:2210.05061, 2022,
  not 2025) is by Gallego-Mejia, Bustos-Brinez and González; AD-DMKDE
  (arXiv:2210.14796) is by Bustos-Brinez, Gallego-Mejia and González.
- Documented that the score averages the squared kernel (effective bandwidth
  σ/√2) and is an unnormalized density estimate.
- Removed `SUBMISSION.md`, the 0.1.0 competition write-up, and its README
  link: it repeated claims this release corrects (a hardware Qiskit path,
  zero-copy interop, "wins both metrics" on KDD, a wrong author order in its
  BibTeX). It remains in the git history.
- Removed false claims: OpenMP parallelism (there are no pragmas),
  "zero-copy" numpy interop, and "O(1)-per-sample" streaming (the true cost is
  O(D²) time and O(1) extra memory per update). Qiskit docs no longer claim
  `StatevectorEstimator`, `DensityMatrix` or a hardware path, and
  BENCHMARKS.md no longer claims a "1:1 parameterised circuit for hardware
  offload" (the backend is statevector simulation only).
- Benchmark wording corrected (numbers unchanged): "wins outright" removed;
  the tables are stated to be one seed-42 split with σ hand-picked on that
  split; the KDD result is numeric-only (categorical columns dropped), not a
  mixed-feature-type result; seed variation added (KDD seeds 1-2:
  OneClassSVM best; ring seeds 0-9: LOF best on every seed; the previously
  omitted `sklearn` scenario: LOF best, DMKDE never); "no hyperparameter
  search" now notes the σ sensitivity. README numbers now come from the same
  Python harness as BENCHMARKS.md (ring 0.962, not the C++ demo's 0.836;
  bimodal Mahalanobis 0.436; 300 streaming updates, from `example.py`).
- Documented `partial_fit`'s forgetting (memory ≈ 1/α rows), the threshold,
  σ and noise-floor behaviour, the PSD acceptance boundary and the raw
  `Mahalanobis::score` -inf case. `example.py` calibrates on held-out data.

**Release pipeline** (`.github/workflows/wheels.yml`)
- `permissions: contents: read` by default; every action pinned to a full
  commit SHA; publishing uses PyPI Trusted Publishing (`id-token: write`,
  `environment: pypi`) and the workflow reads no API token. The old
  `PYPI_API_TOKEN` repository secret has been deleted, and the `pypi`
  environment has a required reviewer and only allows `v*` tags to deploy.
- The tag must equal the version in `pyproject.toml`,
  `python/dmkde/__init__.py` (`__version__`) and `CITATION.cff`; the full
  pytest suite runs against every built wheel and against the installed sdist
  before anything is published; `tests/` ships in the sdist.

### Behaviour changes (may need code changes)

- **`decision_function` now returns `score_samples(X) - offset_`** (sklearn
  convention: negative ⇒ outlier, `predict == where(decision_function >= 0,
  1, -1)`). In 0.1.0 it returned the raw score. Use `score_samples` for the
  raw score.
- **New `contamination` parameter (default 0.05): the target false-alarm
  rate on fresh normal data.** `fit` sets `offset_` from the exact
  **leave-one-out** training scores, so `predict` works right after `fit`
  (0.1.0 raised until `calibrate` was called). `offset_` is the k-th
  smallest of the N leave-one-out scores with `k = floor((N + 1)·c)`: if a
  fresh score and the N scores are exchangeable, the fresh score's rank is
  uniform on 1..N+1, so it falls below the k-th smallest with probability
  `k/(N + 1) <= c`. The expected fraction of fresh normal data flagged is
  therefore at most `c` for every N, and about `c` for large N (0.04995 at
  N = 1000, c = 0.05). (Held-out scores are exactly exchangeable with fresh
  ones; leave-one-out scores average over N − 1 rows instead of N, so for
  them the bound is approximate to O(1/N) and checked by Monte Carlo: mean
  fresh flag rate 0.043-0.047 at N = 20, 0.048-0.050 at N = 200 and 1000.)
  The release candidate's `np.quantile` interpolates above that order
  statistic for small N and flagged on average 0.37-0.44 of fresh data at
  N = 2, 0.17-0.31 at N = 5 and 0.07-0.13 at N = 20. **Small-N caveat:**
  when `k = 0` (N + 1 < 1/c, i.e. N < 19 at c = 0.05) no threshold can keep
  the rate at `c`, so `offset_ = 0.0` (scores are >= 0: nothing is flagged)
  and `fit` warns; when `N·c < 5` the rate of a single fit is noisy (only
  its average is bounded) and `fit` warns. `calibrate` uses the same order
  statistic on the held-out scores (and the same warnings). Leave-one-out
  scores are clipped to the in-sample score (true in exact arithmetic):
  on all-duplicate training data rounding put them an ulp above it, so
  `offset_` exceeded every score and `predict` flagged 100% of the training
  rows. Each in-sample training score
  contains its own row's term, which is exactly `1/N` because RFF embeddings
  have unit norm; removing it gives `s_loo = (N·s − 1)/(N − 1)`, the score of
  the row under the model refitted without it (verified against explicit
  refits to 1e-13). The release candidate used the in-sample quantile, and at
  the default σ with d ≥ 8 standardized features `predict` then flagged
  49-100% of fresh normal data instead of 5%. With a single training row `offset_ = 0.0` (nothing is
  flagged) and `fit` warns. `score_samples` is unchanged. Consequence:
  `predict` on the training set itself flags fewer than `contamination` of
  it, so `check_estimator` fails `check_outliers_train` and
  `check_outliers_fit_predict` (the tests pass them as expected failures,
  as scikit-learn exempts `LocalOutlierFactor(novelty=True)`).
  `fit_predict` stays `fit(X).predict(X)`: thresholding the leave-one-out
  training scores there would flag about `c` of X, but
  `check_outliers_fit_predict` first asserts `fit_predict(X) ==
  fit(X).predict(X)`, so it would still fail, and the two would disagree. `fit` scores
  the training set once, 2-6× the `fit` time of 0.1.0 in our measurements.
- PyOD `DMKDEDetector`: `decision_scores_`, `threshold_` and `labels_` come
  from the same leave-one-out training scores (as PyOD's KNN / LOF exclude
  each point from its own neighbourhood), and `threshold_ = -offset_` uses
  the same order statistic instead of PyOD's percentile (which over-flags at
  small N the same way); `labels_` marks `k − 1` training rows. Fresh-data
  false alarms at `contamination=0.05`, D = 256, d = 8, σ = 1: about 5%
  (release candidate: 50%).
- **Noise-floor warning.** `fit` emits a `UserWarning` when the median
  leave-one-out training score is below `3/D`. Far-apart points still have a
  squared RFF overlap of about `1/D`, so such scores are mostly noise.
  Measured against the exact squared-kernel density on standardized
  Gaussian data (d = 2-100, D = 256/1024) **at N = 500**: median·D ≤ 1.01
  always gave AUC 0.45-0.56, median·D 1.1-1.5 lost 0.07-0.18 AUC, 1.5-2.1
  lost 0.01-0.02, 3-4 at most 0.02 and above 10 at most 0.006. These
  numbers do not carry over to small N: at **N = 10** fits with median·D
  3-6, which do not warn, still lost up to 0.09 AUC (0.07-0.09; N = 30:
  0.04-0.05; N = 100: 0.015-0.025). The message states the ratio to
  `1/feature_dim` and names both causes (σ too small, or `feature_dim` too
  small, e.g. D = 6 for Qiskit). The docstring and README now say to
  standardize features and grow σ with d (σ ≈ sqrt(d), or, after
  standardizing, the median pairwise distance / sqrt(2)).
- `calibrate(X, contamination=None)` sets `offset_`; `contamination` defaults
  to the estimator's parameter. The private `_default_threshold_` attribute is
  gone.
- `partial_fit(X, y=None, alpha=0.01)`: `alpha` must be passed by keyword
  (`partial_fit(X, 0.1)` raises `TypeError`; passing `y=None` explicitly
  and alpha as a third positional argument is accepted). On an unfitted
  estimator the first call fits.
- 1-D input to `score_samples` / `predict` / `decision_function` raises
  instead of being read as one row.
- Invalid input (wrong width, NaN / inf, 0 features, bad parameters) raises
  `ValueError` instead of returning garbage. Unfitted use raises
  `NotFittedError`.
- `roc_auc` raises `ValueError` for empty, non-1-D or non-finite input (0.1.0
  returned 0.5 for empty input and an order-dependent value for NaN). The C++
  `roc_auc` throws `std::invalid_argument` for the same cases.
- Mahalanobis ridge is per-feature relative, `C + 1e-6 · diag(C)`, instead of
  a fixed `1e-6 · I`, so the distance is invariant to rescaling each feature
  independently (tested with factors 1e-4 to 1e4: scores equal to 1e-9
  relative, identical ranking and AUC; the release candidate's
  `1e-6 · trace(C)/d` ridge was invariant only to a common scale, and with
  one feature 1e4 times smaller than the others its AUC fell from 0.971 to
  0.522). That invariance holds for non-constant features. **Constant
  features** (every training value identical, min == max) have no scale of their
  own and use an **absolute** one: inverse variance `1e6`, exactly what
  0.1.0's `1e-6 · I` ridge gave, independent of every other column, so a
  deviation δ alone is distance `1000·|δ|`. (The 0.1.1 release candidate
  used `1e-6` × the mean variance of the *other* columns, so a training
  constant flag that changed at test time was detected with AUC 1.000 when
  the other columns had std 1 but 0.65 at std 1e3 and 0.50 at 1e5 and 1e7;
  now 1.000 at all four.) Scores along a constant feature are therefore not
  invariant to rescaling that feature itself: a 0.01 deviation scores -10,
  and -10⁴ after multiplying the column by 1000. A column with min == max is
  treated as exactly constant with its exact value as the mean, and other
  means get one refinement pass. (A later release candidate summed the mean
  naively: a constant such as 0.1 or a timestamp, which is not exactly
  summable, drifted by ~n·eps, so at n ≥ 1000 the column got inverse variance
  ~1e29 instead of 1e6 and a 1e-6 deviation scored -7e8; a constant 1e300
  column was rejected as "too large".) Any column with two distinct values is scaled by its own measured
  variance, however small relative to its magnitude: a later candidate also
  classed spreads under ~64 ulp as constant, so 1e14 + N(0,1) noise, ns
  timestamps or 1 + 1e-14·N(0,1) got the absolute 1e6 scale and AUC near 0.5
  (1.0 after shifting the column to zero). The mean is stored to the column's own
  precision, so a column whose spread is only a few ulp of its magnitude still
  scores off by up to ~0.5·ulp/std along it; subtract a reference value first. Data where every feature is
  constant is rejected. The covariance is now computed on each column
  divided by a power of two near its largest magnitude (an exact rescaling),
  so tiny or huge units no longer underflow or overflow: a feature with
  standard deviation below ~1e-154 or above ~1e154, whose inverse variance
  cannot be stored in double precision, is rejected with a `ValueError`
  naming the feature and saying "underflowed" / "overflowed" (the release
  candidate called a 1e-160-scale feature "training data too large" and
  silently treated a 1e-170-scale one as constant, scoring 84% off).
  Re-measured against 0.1.0: standardized and correlated unit-variance data
  agree to within ~1e-7 relative; near-collinear features, where the ridge
  dominates the near-singular direction, differ by up to ~2e-2; on features
  whose variances are far from 1, 0.1.0's absolute ridge distorted the
  distance (by ~5e-3 relative with variances 1e-4 to 1e6), which 0.1.1
  removes. The synthetic-benchmark Mahalanobis AUCs (`benchmark.py`, seed
  42) reproduce to the 4 printed decimals; the KDD and credit-card
  Mahalanobis rows were not re-run (their inputs are standardized, where the
  ridge change moves scores by ~1e-7 relative).
- C++: constructors validate parameters; `DMKDE` gains an optional
  `allow_large` constructor argument; `score` / `update` before `fit` throw
  `std::logic_error`. The checked `score` / `score_batch` overloads throw
  `std::invalid_argument` for a finite row whose embedding overflows (the raw
  pointer-only `score(x)` still returns NaN for it; the raw
  `Mahalanobis::score(x)` returns NaN for a non-finite `x - mean` and -inf
  for a distance beyond DBL_MAX). `DMKDE::from_state` and
  `Mahalanobis::set_state` reject non-PSD ρ / non-SPD `inv_cov`.
- **Loading cost.** For D ≤ 2048 (`kFullPsdCheckMaxDim`) the PSD check is a
  full shifted Cholesky factorisation, now blocked with a register-tiled
  trailing update. Pickle load times on a 4-core laptop (one core), release
  candidate → 0.1.1: D = 1024 0.26-0.69 s → 0.07 s, D = 2048 2.3-5.0 s →
  0.4-0.5 s, D = 4096 19-37 s → 1.2-1.5 s, D = 8192 454 s → 10-13 s. Above
  D = 2048 a Cholesky would still cost 2.3 s (4096) to 21 s (8192), so ρ gets
  O(D²) necessary checks (non-negative diagonal, every 2×2 principal minor)
  plus a randomised search for a vector v with vᵀρv < -1e-9·trace·‖v‖² (4
  Gaussian probes and 32 Lanczos steps with full reorthogonalisation, whose
  smallest Ritz vector is re-checked exactly). Every rejection is backed by
  such a v, so valid states are never rejected, but above D = 2048 a negative
  eigenvalue that is small relative to the spread of ρ's spectrum can be
  missed; states produced by this library are PSD by construction. The
  probe vectors come from a fresh seed per call (`std::random_device` mixed
  with a call counter and the clock); the release candidate used a fixed
  public seed, so a state with an eigenvalue of -1e-4 (1e5 × the tolerance)
  along a direction orthogonal to its known probes loaded deterministically.
  The D > 2048 path is still **probabilistic and not adversarially robust**
  (e.g. an eigenvalue of -1e-6 there can pass); load such states only from
  trusted sources. D ≤ 2048 stays an exact Cholesky check.

### Not in this release (planned for 0.2.0)

- A data-dependent default `sigma`. The default `sigma=1.0` is kept in 0.1.1
  (changing it would change every default model); it suits only a few
  standardized features, and above ~8 features it puts the scores at the
  random-feature noise floor, which 0.1.1 detects and warns about.
- InQMAD threshold-gated streaming update (updates remain unconditional).
- The paper's `1/M_γ` density normalization.
- A benchmark program with multiple splits and seeds, held-out selection of
  σ, and confidence intervals. 0.1.1 only corrects the BENCHMARKS.md
  wording: the published numbers are one seed-42 split with σ chosen on
  that split, and other seeds change the winners (KDD seeds 1-2: OneClassSVM
  best; ring seeds 0-9 and the `sklearn` scenario: LOF best).
- Identical models across platforms. The random Fourier weights are drawn with
  `std::normal_distribution`, whose output is implementation-defined, so the same
  `random_state` gives a different model on Windows (MSVC) than on Linux/macOS
  (confirmed in CI). Pickled models are unaffected: they store the weights. 0.2.0 will
  draw the weights with a portable generator, which changes scores for a given seed.
