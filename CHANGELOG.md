# Changelog

## 0.1.1 — safety release

Fixes the flaws found by two independent audits of 0.1.0. For valid input
`DMKDE.score_samples` / `_core.DMKDE.score_batch` return bit-identical
values to 0.1.0 (same seed ⇒ same RFF draw, same accumulation order), and
the published seed-42 benchmark numbers reproduce unchanged.

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
  `DMKDE` and `Mahalanobis` pass `sklearn.utils.estimator_checks.check_estimator`.
- `random_state` accepts None, an int, a `RandomState` or a `Generator`. An
  int is used as the C++ seed directly, so `random_state=42` gives the same
  model as 0.1.0.
- Pickling: `DMKDE` and `Mahalanobis` pickle via `__getstate__` /
  `__setstate__`. The state stores the RFF weights `W`, ρ and the counters
  (not just the seed), so a pickle loads identically on any platform. New
  `_core.DMKDE.get_state()` / `from_state()` (and Mahalanobis equivalents)
  validate every field on import: `W` and ρ finite, ρ symmetric with trace 1
  and positive semidefinite (smallest eigenvalue ≥ -1e-9·trace, checked by a
  shifted Cholesky factorisation, O(D³)); Mahalanobis `mean` finite and
  `inv_cov` finite, symmetric and positive definite. Malformed states (not a
  dict, missing keys, wrong types such as a float `format` or a string
  `fitted`, negative `seed`, out-of-range dimensions or counters, non-numeric
  arrays) raise `ValueError`, not a `RuntimeError` cast error.
- `fit` resets any threshold set by `calibrate`.
- `_core.DMKDE.fit` resets `n_streamed`.
- PyOD adapter: `DMKDEDetector` is one cached module-level class, so
  `isinstance` and pickling work; `predict_proba` works; an unfitted
  `decision_function` raises `NotFittedError`; input width is checked.
- Qiskit backend: `encode` checks width and finiteness; ρ is re-read from the
  model whenever it changed (it was a stale snapshot taken at construction).
  The power-of-two check and the qubit count use the fitted D
  (`model._impl.feature_dim`), not the `feature_dim` parameter, which
  `set_params` can change after fit; a model refit to a different D after
  wrapping raises `ValueError` instead of being mis-simulated.

**GIL**
- `fit`, `score_batch`, `update`, Mahalanobis `fit` / `score_batch` and
  `roc_auc` release the GIL. Each model has a writer-preferring reader/writer
  lock (mutex + condition variable), so scoring runs in parallel across
  threads while `fit` / `update` are exclusive, and once a writer is waiting
  new readers queue behind it: continuous scoring can no longer delay
  `update` / `partial_fit` for seconds (`std::shared_mutex` in libstdc++
  prefers readers).

**Docs and metadata**
- Citations corrected everywhere: arXiv:2201.10006 is by Useche,
  Bustos-Brinez, Gallego-Mejia and González; InQMAD (arXiv:2210.05061, 2022,
  not 2025) is by Gallego-Mejia, Bustos-Brinez and González; AD-DMKDE
  (arXiv:2210.14796) is by Bustos-Brinez, Gallego-Mejia and González.
- Documented that the score averages the squared kernel (effective bandwidth
  σ/√2) and is an unnormalized density estimate.
- Removed false claims: OpenMP parallelism (there are no pragmas),
  "zero-copy" numpy interop, and "O(1)-per-sample" streaming (the true cost is
  O(D²) time and O(1) extra memory per update). Qiskit docs no longer claim
  `StatevectorEstimator`, `DensityMatrix` or a hardware path.

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
- **New `contamination` parameter (default 0.05).** `fit` sets `offset_` to
  that quantile of the training scores, so `predict` works right after `fit`
  (0.1.0 raised until `calibrate` was called). This scores the training set
  once, roughly doubling `fit` time for large N·D².
- `calibrate(X, contamination=None)` sets `offset_`; `contamination` defaults
  to the estimator's parameter. The private `_default_threshold_` attribute is
  gone.
- `partial_fit(X, y=None, alpha=0.01)`: `alpha` must be passed by keyword
  (passing a scalar positionally raises `TypeError`). On an unfitted
  estimator the first call fits.
- 1-D input to `score_samples` / `predict` / `decision_function` raises
  instead of being read as one row.
- Invalid input (wrong width, NaN / inf, 0 features, bad parameters) raises
  `ValueError` instead of returning garbage. Unfitted use raises
  `NotFittedError`.
- `roc_auc` raises `ValueError` for empty, non-1-D or non-finite input (0.1.0
  returned 0.5 for empty input and an order-dependent value for NaN). The C++
  `roc_auc` throws `std::invalid_argument` for the same cases.
- Mahalanobis ridge is `1e-6 · trace(C)/d` instead of a fixed `1e-6`, making
  the score scale-invariant. For standardized data (trace(C)/d ≈ 1) results
  are unchanged to within ~1e-9 relative; up to ~1e-3 on near-collinear
  features, where the ridge matters. The published benchmark numbers
  reproduce.
- C++: constructors validate parameters; `DMKDE` gains an optional
  `allow_large` constructor argument; `score` / `update` before `fit` throw
  `std::logic_error`. The checked `score` / `score_batch` overloads throw
  `std::invalid_argument` for a finite row whose embedding overflows (the raw
  pointer-only `score(x)` still returns NaN for it). `DMKDE::from_state` and
  `Mahalanobis::set_state` reject non-PSD ρ / non-SPD `inv_cov`, so loading a
  model costs O(D³) (a few ms at the default D = 256).

### Not in this release (planned for 0.2.0)

- InQMAD threshold-gated streaming update (updates remain unconditional).
- The paper's `1/M_γ` density normalization.
- Benchmark re-runs and changes to the BENCHMARKS.md results.
- Identical models across platforms. The random Fourier weights are drawn with
  `std::normal_distribution`, whose output is implementation-defined, so the same
  `random_state` gives a different model on Windows (MSVC) than on Linux/macOS
  (confirmed in CI). Pickled models are unaffected: they store the weights. 0.2.0 will
  draw the weights with a portable generator, which changes scores for a given seed.
