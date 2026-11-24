// dmkde.hpp — Density Matrix Kernel Density Estimation (single-header)
//
// Implementation of the density-matrix kernel density estimator described in:
//
//   D. H. Useche, O. A. Bustos-Brinez, J. A. Gallego-Mejia, F. A. González,
//   "Quantum density estimation with density matrices:
//    Application to quantum anomaly detection."
//   Phys. Rev. A 109, 032418 (2024).   arXiv:2201.10006
//
// related anomaly-detection work by the same group:
//
//   O. Bustos-Brinez, J. Gallego-Mejia, F. A. González,
//   "AD-DMKDE: Anomaly Detection through Density Matrices and Fourier
//    Features."  arXiv:2210.14796 (2022).
//
// and the streaming variant in:
//
//   J. Gallego-Mejia, O. Bustos-Brinez, F. A. González,
//   "InQMAD: Incremental Quantum Measurement Anomaly Detection."
//   arXiv:2210.05061 (2022).
//
// Algorithm in one paragraph:
//   Each input x ∈ R^d is embedded into R^D by Random Fourier Features
//   (RFF), which approximates a Gaussian kernel as ⟨φ(x), φ(y)⟩ ≈ k(x,y)
//   with k(x, y) = exp(-‖x - y‖² / (2σ²)).
//   Training builds the empirical density operator
//
//       ρ = (1/N) Σ_i  φ(x_i) φ(x_i)^T     (PSD, trace 1)
//
//   For a new x, the Born-rule score
//
//       s(x) = φ(x)^T  ρ  φ(x)  =  (1/N) Σ_i ⟨φ(x), φ(x_i)⟩²   ∈ [0, 1]
//
//   is a mean of the SQUARED kernel, k(x, x_i)² = exp(-‖x - x_i‖² / σ²),
//   i.e. a Gaussian with effective bandwidth σ/√2, not σ. The score is an
//   UNNORMALIZED kernel density estimate: it does not integrate to 1 (the
//   paper's 1/M_γ normalization is not applied). It ranks points by density,
//   which is all an anomaly detector needs. Low s ⇒ anomalous.
//
//   The RFF frequencies are a fixed, seeded draw ω ~ N(0, σ⁻² I); they are
//   not learned/optimised (the class keeps its historical name
//   AdaptiveFourierFeatures).
//
// Streaming: ρ_{t+1} = (1-α) ρ_t + α · φ(x_t) φ(x_t)^T   (rank-1 EMA).
//   Cost per update: O(D²) time and O(1) extra memory beyond ρ (O(D)
//   scratch). Updates are unconditional (InQMAD's threshold gating is not
//   implemented).
//
// API safety:
//   * Functions taking `const double*` WITHOUT a length (score(x),
//     update(x, a), transform(x, out), Mahalanobis::score(x)) are the raw,
//     UNCHECKED fast path: the caller guarantees the buffer holds
//     input_dim() doubles (and transform's `out` holds feature_dim()).
//   * Every other entry point (std::vector overloads and pointer+length
//     overloads) checks sizes and throws std::invalid_argument on mismatch.
//   * All fit / update paths reject NaN / ±inf (std::invalid_argument).
//   * Using a model before fit() throws std::logic_error.
//
// Threading: const member functions may run concurrently with each other.
//   Mutating calls (fit, update) need external synchronisation against any
//   other call on the same object. The Python bindings provide that lock.
//
// This is a pure C++17 header-only library with no external dependencies.
// All loops are plain single-threaded C++ (no OpenMP).
//
// License: MIT.  See LICENSE.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dmkde {

constexpr double PI = 3.14159265358979323846;

// Default cap on the RFF dimension D. ρ is a dense D×D matrix of doubles,
// so D = 2^14 already needs 2 GiB. Larger D must be requested explicitly
// with allow_large = true.
constexpr int kMaxFeatureDim = 1 << 14;

namespace detail {

inline bool all_finite(const double* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

inline void check_sigma(double sigma) {
    if (!std::isfinite(sigma) || !(sigma > 0.0) || !std::isfinite(1.0 / sigma))
        throw std::invalid_argument("sigma must be finite and > 0");
}

inline void check_dims(int input_dim, int feature_dim, bool allow_large) {
    if (input_dim <= 0)
        throw std::invalid_argument("input_dim must be > 0");
    if (feature_dim <= 0 || feature_dim % 2 != 0)
        throw std::invalid_argument("feature_dim must be positive and even");
    if (feature_dim > kMaxFeatureDim && !allow_large)
        throw std::invalid_argument(
            "feature_dim=" + std::to_string(feature_dim) + " exceeds the default cap of " +
            std::to_string(kMaxFeatureDim) + " (rho needs feature_dim^2 * 8 bytes); "
            "pass allow_large=true to opt in");
    if ((size_t)feature_dim > std::numeric_limits<size_t>::max() / (size_t)feature_dim)
        throw std::length_error("feature_dim^2 overflows size_t");
}

}  // namespace detail

// =====================================================================
//  AdaptiveFourierFeatures
//  ------------------------
//  Maps R^d → R^D via stacked cos/sin projections of d-dim Gaussian
//  random frequencies. The kernel approximated is the Gaussian / RBF
//
//      k(x, y) = exp( -||x - y||^2 / (2 σ^2) )
//
//  This is the RFF map of Rahimi & Recht (NIPS 2007). Despite the name,
//  the frequencies are a fixed seeded draw, not learned.
// =====================================================================
class AdaptiveFourierFeatures {
public:
    AdaptiveFourierFeatures() = default;

    void fit(int input_dim, int output_dim, double sigma, uint64_t seed) {
        if (input_dim <= 0)
            throw std::invalid_argument("input_dim must be > 0");
        if (output_dim <= 0 || output_dim % 2 != 0)
            throw std::invalid_argument("output_dim must be positive and even (cos+sin pairs)");
        detail::check_sigma(sigma);
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        sigma_      = sigma;
        const int K = output_dim / 2;

        std::mt19937_64 rng(seed);
        // ω ~ N(0, σ^-2 I)  → variance 1/σ^2, std 1/σ
        std::normal_distribution<double> gauss(0.0, 1.0 / sigma);
        W_.assign((size_t)K * input_dim, 0.0);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < input_dim; ++j)
                W_[(size_t)k * input_dim + j] = gauss(rng);

        scale_ = std::sqrt(2.0 / (double)output_dim);
    }

    // Install explicit frequencies W (K × input_dim row-major, K = output_dim/2).
    // Used to restore a saved model bit-for-bit on any platform (the
    // std::normal_distribution sequence is implementation-defined).
    void set_weights(int input_dim, int output_dim, double sigma,
                     const std::vector<double>& W) {
        if (input_dim <= 0)
            throw std::invalid_argument("input_dim must be > 0");
        if (output_dim <= 0 || output_dim % 2 != 0)
            throw std::invalid_argument("output_dim must be positive and even (cos+sin pairs)");
        detail::check_sigma(sigma);
        if (W.size() != (size_t)(output_dim / 2) * (size_t)input_dim)
            throw std::invalid_argument("W must have (output_dim/2) * input_dim entries");
        if (!detail::all_finite(W.data(), W.size()))
            throw std::invalid_argument("W contains NaN or inf");
        input_dim_  = input_dim;
        output_dim_ = output_dim;
        sigma_      = sigma;
        W_          = W;
        scale_      = std::sqrt(2.0 / (double)output_dim);
    }

    // UNCHECKED: x must hold input_dim() doubles, out must hold output_dim().
    void transform(const double* x, double* out) const {
        const int K = output_dim_ / 2;
        for (int k = 0; k < K; ++k) {
            double dot = 0;
            const double* w = &W_[(size_t)k * input_dim_];
            for (int j = 0; j < input_dim_; ++j) dot += w[j] * x[j];
            out[2 * k]     = scale_ * std::cos(dot);
            out[2 * k + 1] = scale_ * std::sin(dot);
        }
    }

    // Size-checked transform.
    void transform(const double* x, size_t n, double* out, size_t out_n) const {
        if (output_dim_ == 0) throw std::logic_error("AdaptiveFourierFeatures used before fit()");
        if (n != (size_t)input_dim_)
            throw std::invalid_argument("x has " + std::to_string(n) + " features, expected " +
                                        std::to_string(input_dim_));
        if (out_n != (size_t)output_dim_)
            throw std::invalid_argument("output buffer has wrong size");
        transform(x, out);
    }

    int input_dim()  const { return input_dim_; }
    int output_dim() const { return output_dim_; }
    double sigma()   const { return sigma_; }
    const std::vector<double>& weights() const { return W_; }

private:
    int input_dim_  = 0;
    int output_dim_ = 0;
    double sigma_   = 1.0;
    double scale_   = 1.0;
    std::vector<double> W_;  // (K, input_dim) row-major; K = output_dim / 2
};

// Complete serialisable state of a DMKDE model.
struct DMKDEState {
    int input_dim   = 0;
    int feature_dim = 0;
    double sigma    = 1.0;
    uint64_t seed   = 0;          // informational; W below is authoritative
    std::vector<double> W;        // (feature_dim/2) × input_dim, row-major
    std::vector<double> rho;      // feature_dim × feature_dim, row-major
    long long n_fitted   = 0;
    long long n_streamed = 0;
    bool fitted = false;
};

// =====================================================================
//  DMKDE
//  -----
//  Density-matrix kernel density estimator. Stores ρ ∈ R^{D×D} as a
//  dense symmetric matrix, fit by accumulating outer products.
// =====================================================================
class DMKDE {
public:
    // Build a model with a Gaussian RFF embedding into R^D.
    // Throws std::invalid_argument for input_dim <= 0, odd / non-positive
    // feature_dim, feature_dim > kMaxFeatureDim (unless allow_large), or
    // sigma that is not finite and > 0.
    DMKDE(int input_dim, int feature_dim, double sigma, uint64_t seed = 42,
          bool allow_large = false)
        : D_(validated_D(input_dim, feature_dim, sigma, allow_large)),
          seed_(seed),
          allow_large_(allow_large),
          rho_((size_t)D_ * D_, 0.0) {
        aff_.fit(input_dim, feature_dim, sigma, seed);
    }

    // Batch fit: rho = (1/N) Σ φ(x_i) φ(x_i)^T.
    // Only the upper triangle is computed and then mirrored.
    // Resets the streaming counter. Rows must have input_dim() entries
    // and be finite.
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty()) throw std::invalid_argument("empty training set");
        const size_t d_in = (size_t)aff_.input_dim();
        for (const auto& row : X) {
            if (row.size() != d_in)
                throw std::invalid_argument("training row dimension mismatch");
            if (!detail::all_finite(row.data(), row.size()))
                throw std::invalid_argument("training data contains NaN or inf");
        }
        fit_impl(X.size(), [&](size_t i) { return X[i].data(); });
    }

    // Batch fit from a contiguous row-major (n_rows × n_cols) buffer.
    void fit(const double* X, size_t n_rows, size_t n_cols) {
        if (n_rows == 0) throw std::invalid_argument("empty training set");
        check_cols(n_cols);
        if (!detail::all_finite(X, n_rows * n_cols))
            throw std::invalid_argument("training data contains NaN or inf");
        fit_impl(n_rows, [&](size_t i) { return X + i * n_cols; });
    }

    // Streaming rank-1 EMA update (InQMAD-style, unconditional).
    // ρ_{t+1} = (1 - α) ρ_t + α φ(x) φ(x)^T.
    // O(D²) time per update; O(1) extra memory beyond ρ.
    // UNCHECKED width: x must hold input_dim() doubles. Rejects non-finite
    // x and α ∉ (0, 1]; throws std::logic_error before fit(). On any throw
    // ρ is unchanged.
    void update(const double* x, double alpha) {
        require_fitted("update");
        if (!(alpha > 0.0 && alpha <= 1.0))
            throw std::invalid_argument("alpha must be in (0, 1]");
        if (!detail::all_finite(x, (size_t)aff_.input_dim()))
            throw std::invalid_argument("x contains NaN or inf");
        std::vector<double> phi(D_);
        aff_.transform(x, phi.data());
        if (!detail::all_finite(phi.data(), phi.size()))
            throw std::invalid_argument("RFF embedding of x is not finite (input too large)");
        const double oneMinusA = 1.0 - alpha;
        for (int i = 0; i < D_; ++i) {
            const double phi_i = phi[i];
            double* row = &rho_[(size_t)i * D_];
            for (int j = i; j < D_; ++j) {
                row[j] = oneMinusA * row[j] + alpha * phi_i * phi[j];
                rho_[(size_t)j * D_ + i] = row[j];
            }
        }
        ++n_streamed_;
        ++revision_;
    }

    // Size-checked update.
    void update(const double* x, size_t n, double alpha) {
        check_cols(n);
        update(x, alpha);
    }
    void update(const std::vector<double>& x, double alpha) {
        update(x.data(), x.size(), alpha);
    }

    // Born-rule density score: s(x) = φ(x)^T ρ φ(x).
    // Higher = more "normal"; lower = more anomalous.
    // UNCHECKED width: x must hold input_dim() doubles. A non-finite x
    // yields NaN. Throws std::logic_error before fit().
    double score(const double* x) const {
        require_fitted("score");
        std::vector<double> phi(D_), rphi(D_, 0.0);
        aff_.transform(x, phi.data());
        // rphi = ρ φ
        for (int i = 0; i < D_; ++i) {
            const double* row = &rho_[(size_t)i * D_];
            double s = 0;
            for (int j = 0; j < D_; ++j) s += row[j] * phi[j];
            rphi[i] = s;
        }
        // s = φ^T rphi
        double s = 0;
        for (int i = 0; i < D_; ++i) s += phi[i] * rphi[i];
        return s;
    }

    // Size-checked score; also rejects non-finite x.
    double score(const double* x, size_t n) const {
        check_cols(n);
        if (!detail::all_finite(x, n))
            throw std::invalid_argument("x contains NaN or inf");
        return score(x);
    }
    double score(const std::vector<double>& x) const { return score(x.data(), x.size()); }

    // Size-checked batch score over a row-major (n_rows × n_cols) buffer.
    // out must hold n_rows doubles. Rejects non-finite input.
    void score_batch(const double* X, size_t n_rows, size_t n_cols, double* out) const {
        require_fitted("score_batch");
        check_cols(n_cols);
        if (!detail::all_finite(X, n_rows * n_cols))
            throw std::invalid_argument("X contains NaN or inf");
        for (size_t i = 0; i < n_rows; ++i) out[i] = score(X + i * n_cols);
    }

    int feature_dim() const { return D_; }
    int input_dim()   const { return aff_.input_dim(); }
    double sigma()    const { return aff_.sigma(); }
    uint64_t seed()   const { return seed_; }
    bool allow_large() const { return allow_large_; }
    bool fitted()     const { return fitted_; }
    long long n_fitted()   const { return n_fitted_; }
    long long n_streamed() const { return n_streamed_; }
    // Incremented on every change of ρ (fit, update, state import).
    uint64_t revision() const { return revision_; }

    // Direct accessors for backends that need raw state.
    const std::vector<double>& rho() const { return rho_; }
    const std::vector<double>& weights() const { return aff_.weights(); }

    // UNCHECKED: x must hold input_dim() doubles, out feature_dim() doubles.
    void transform(const double* x, double* out) const {
        aff_.transform(x, out);
    }
    // Size-checked transform.
    std::vector<double> transform(const std::vector<double>& x) const {
        std::vector<double> out(D_);
        aff_.transform(x.data(), x.size(), out.data(), out.size());
        return out;
    }
    void transform(const double* x, size_t n, double* out, size_t out_n) const {
        aff_.transform(x, n, out, out_n);
    }

    double trace() const {
        double t = 0;
        for (int i = 0; i < D_; ++i) t += rho_[(size_t)i * D_ + i];
        return t;
    }

    // ---- serialisation -------------------------------------------------
    DMKDEState state() const {
        DMKDEState s;
        s.input_dim   = input_dim();
        s.feature_dim = D_;
        s.sigma       = aff_.sigma();
        s.seed        = seed_;
        s.W           = aff_.weights();
        s.rho         = rho_;
        s.n_fitted    = n_fitted_;
        s.n_streamed  = n_streamed_;
        s.fitted      = fitted_;
        return s;
    }

    // Rebuild a model from a saved state. Validates shapes, finiteness,
    // symmetry and trace of ρ, and the counters; throws
    // std::invalid_argument on any inconsistency.
    static DMKDE from_state(const DMKDEState& s, bool allow_large = false) {
        DMKDE m(s.input_dim, s.feature_dim, s.sigma, s.seed, allow_large);
        m.aff_.set_weights(s.input_dim, s.feature_dim, s.sigma, s.W);
        const size_t D = (size_t)s.feature_dim;
        if (s.rho.size() != D * D)
            throw std::invalid_argument("rho must have feature_dim^2 entries");
        if (!detail::all_finite(s.rho.data(), s.rho.size()))
            throw std::invalid_argument("rho contains NaN or inf");
        if (s.n_fitted < 0 || s.n_streamed < 0)
            throw std::invalid_argument("counters must be non-negative");
        if (s.fitted) {
            if (s.n_fitted < 1)
                throw std::invalid_argument("a fitted state needs n_fitted >= 1");
            double tr = 0;
            for (size_t i = 0; i < D; ++i) {
                tr += s.rho[i * D + i];
                for (size_t j = i + 1; j < D; ++j) {
                    const double a = s.rho[i * D + j], b = s.rho[j * D + i];
                    if (std::fabs(a - b) > 1e-12 * (1.0 + std::fabs(a)))
                        throw std::invalid_argument("rho is not symmetric");
                }
            }
            if (std::fabs(tr - 1.0) > 1e-6)
                throw std::invalid_argument("rho must have trace 1");
        } else {
            if (s.n_fitted != 0 || s.n_streamed != 0)
                throw std::invalid_argument("an unfitted state must have zero counters");
            for (double v : s.rho)
                if (v != 0.0) throw std::invalid_argument("an unfitted state must have rho = 0");
        }
        m.rho_        = s.rho;
        m.n_fitted_   = s.n_fitted;
        m.n_streamed_ = s.n_streamed;
        m.fitted_     = s.fitted;
        ++m.revision_;
        return m;
    }

private:
    static int validated_D(int input_dim, int feature_dim, double sigma, bool allow_large) {
        detail::check_dims(input_dim, feature_dim, allow_large);
        detail::check_sigma(sigma);
        return feature_dim;
    }

    void check_cols(size_t n) const {
        if (n != (size_t)aff_.input_dim())
            throw std::invalid_argument("x has " + std::to_string(n) + " features, expected " +
                                        std::to_string(aff_.input_dim()));
    }

    void require_fitted(const char* what) const {
        if (!fitted_)
            throw std::logic_error(std::string("DMKDE::") + what + " called before fit()");
    }

    template <class RowFn>
    void fit_impl(size_t n_rows, RowFn row_at) {
        std::fill(rho_.begin(), rho_.end(), 0.0);
        std::vector<double> phi(D_);
        for (size_t r = 0; r < n_rows; ++r) {
            aff_.transform(row_at(r), phi.data());
            if (!detail::all_finite(phi.data(), phi.size())) {
                // Leave the model in a clean, unfitted state.
                std::fill(rho_.begin(), rho_.end(), 0.0);
                fitted_ = false;
                n_fitted_ = 0;
                n_streamed_ = 0;
                ++revision_;
                throw std::invalid_argument(
                    "RFF embedding of a training row is not finite (input too large)");
            }
            // ρ += φ φ^T  (upper triangle only)
            for (int i = 0; i < D_; ++i) {
                const double phi_i = phi[i];
                double* rho_row = &rho_[(size_t)i * D_];
                for (int j = i; j < D_; ++j) {
                    rho_row[j] += phi_i * phi[j];
                }
            }
        }
        const double inv_n = 1.0 / (double)n_rows;
        for (int i = 0; i < D_; ++i)
            for (int j = i; j < D_; ++j) {
                rho_[(size_t)i * D_ + j] *= inv_n;
                rho_[(size_t)j * D_ + i]  = rho_[(size_t)i * D_ + j];
            }
        n_fitted_   = (long long)n_rows;
        n_streamed_ = 0;
        fitted_     = true;
        ++revision_;
    }

    int D_;
    uint64_t seed_;
    bool allow_large_;
    std::vector<double> rho_;   // D x D row-major
    AdaptiveFourierFeatures aff_;
    long long n_fitted_   = 0;
    long long n_streamed_ = 0;
    bool fitted_          = false;
    uint64_t revision_    = 0;
};

// =====================================================================
//  Mahalanobis baseline
//  --------------------
//  Negative Mahalanobis distance (higher = more "normal").
//  The covariance gets a ridge proportional to its mean variance,
//  C + 1e-6 · (trace(C)/d) · I, so the score is invariant to rescaling
//  the data (a fixed absolute ridge is not).
// =====================================================================
class Mahalanobis {
public:
    // Relative ridge: λ = kRidge · trace(C) / d.
    static constexpr double kRidge = 1e-6;

    // Needs N >= 2 finite rows of equal width d >= 1.
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.size() < 2)
            throw std::invalid_argument("Mahalanobis needs at least 2 training rows");
        const size_t d = X[0].size();
        if (d == 0) throw std::invalid_argument("training rows have 0 features");
        for (const auto& r : X) {
            if (r.size() != d) throw std::invalid_argument("training row dimension mismatch");
            if (!detail::all_finite(r.data(), d))
                throw std::invalid_argument("training data contains NaN or inf");
        }
        fit_impl(X.size(), d, [&](size_t i) { return X[i].data(); });
    }

    // Fit from a contiguous row-major (n_rows × n_cols) buffer.
    void fit(const double* X, size_t n_rows, size_t n_cols) {
        if (n_rows < 2)
            throw std::invalid_argument("Mahalanobis needs at least 2 training rows");
        if (n_cols == 0) throw std::invalid_argument("training rows have 0 features");
        if (!detail::all_finite(X, n_rows * n_cols))
            throw std::invalid_argument("training data contains NaN or inf");
        fit_impl(n_rows, n_cols, [&](size_t i) { return X + i * n_cols; });
    }

    // Returns *negative* Mahalanobis distance so "higher = more normal"
    // matches the DMKDE convention — apples-to-apples for AUC.
    // UNCHECKED width: x must hold input_dim() doubles. Non-finite x
    // yields NaN (never the "most normal" score). Throws before fit().
    double score(const double* x) const {
        if (d_ == 0) throw std::logic_error("Mahalanobis::score called before fit()");
        std::vector<double> diff(d_);
        for (int i = 0; i < d_; ++i) diff[i] = x[i] - mean_[i];
        double q = 0;
        for (int i = 0; i < d_; ++i)
            for (int j = 0; j < d_; ++j)
                q += diff[i] * invCov_[i][j] * diff[j];
        if (std::isnan(q)) return std::numeric_limits<double>::quiet_NaN();
        return -std::sqrt(std::max(0.0, q));
    }

    // Size-checked score; rejects non-finite x.
    double score(const double* x, size_t n) const {
        if (d_ == 0) throw std::logic_error("Mahalanobis::score called before fit()");
        if (n != (size_t)d_)
            throw std::invalid_argument("x has " + std::to_string(n) + " features, expected " +
                                        std::to_string(d_));
        if (!detail::all_finite(x, n)) throw std::invalid_argument("x contains NaN or inf");
        return score(x);
    }
    double score(const std::vector<double>& x) const { return score(x.data(), x.size()); }

    void score_batch(const double* X, size_t n_rows, size_t n_cols, double* out) const {
        if (d_ == 0) throw std::logic_error("Mahalanobis::score_batch called before fit()");
        if (n_cols != (size_t)d_)
            throw std::invalid_argument("X has " + std::to_string(n_cols) + " features, expected " +
                                        std::to_string(d_));
        if (!detail::all_finite(X, n_rows * n_cols))
            throw std::invalid_argument("X contains NaN or inf");
        for (size_t i = 0; i < n_rows; ++i) out[i] = score(X + i * n_cols);
    }

    int input_dim() const { return d_; }
    bool fitted() const { return d_ > 0; }

    // Serialisation: mean (d) and inverse covariance (d × d, row-major).
    const std::vector<double>& mean() const { return mean_; }
    std::vector<double> inv_cov() const {
        std::vector<double> out((size_t)d_ * d_);
        for (int i = 0; i < d_; ++i)
            for (int j = 0; j < d_; ++j) out[(size_t)i * d_ + j] = invCov_[i][j];
        return out;
    }
    void set_state(const std::vector<double>& mean, const std::vector<double>& inv_cov) {
        const size_t d = mean.size();
        if (d == 0) throw std::invalid_argument("mean must be non-empty");
        if (inv_cov.size() != d * d) throw std::invalid_argument("inv_cov must have d*d entries");
        if (!detail::all_finite(mean.data(), d) || !detail::all_finite(inv_cov.data(), d * d))
            throw std::invalid_argument("Mahalanobis state contains NaN or inf");
        std::vector<std::vector<double>> P(d, std::vector<double>(d));
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) P[i][j] = inv_cov[i * d + j];
        mean_ = mean;
        invCov_ = std::move(P);
        d_ = (int)d;
    }

private:
    template <class RowFn>
    void fit_impl(size_t n_rows, size_t n_cols, RowFn row_at) {
        const int d = (int)n_cols;
        std::vector<double> mean(d, 0.0);
        for (size_t r = 0; r < n_rows; ++r) {
            const double* x = row_at(r);
            for (int j = 0; j < d; ++j) mean[j] += x[j];
        }
        for (int j = 0; j < d; ++j) mean[j] /= (double)n_rows;

        std::vector<std::vector<double>> C(d, std::vector<double>(d, 0));
        for (size_t r = 0; r < n_rows; ++r) {
            const double* x = row_at(r);
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    C[i][j] += (x[i] - mean[i]) * (x[j] - mean[j]);
        }
        const double inv = 1.0 / (double)(n_rows - 1);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) C[i][j] *= inv;

        // Scale-relative ridge. Invert C/s + kRidge·I and rescale by 1/s,
        // which equals (C + kRidge·s·I)^-1 with s = trace(C)/d, so the
        // result does not depend on the units of the data.
        double tr = 0;
        for (int i = 0; i < d; ++i) tr += C[i][i];
        const double s = tr / (double)d;
        if (!(s > 0.0) || !std::isfinite(s))
            throw std::invalid_argument(
                "training data has zero (or non-finite) variance in every feature");
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) C[i][j] /= s;
        for (int i = 0; i < d; ++i) C[i][i] += kRidge;
        auto P = invert(C);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) P[i][j] /= s;

        mean_   = std::move(mean);
        invCov_ = std::move(P);
        d_      = d;
    }

    static std::vector<std::vector<double>> invert(std::vector<std::vector<double>> A) {
        const int n = (int)A.size();
        std::vector<std::vector<double>> I(n, std::vector<double>(n, 0));
        for (int i = 0; i < n; ++i) I[i][i] = 1;
        for (int i = 0; i < n; ++i) {
            int piv = i;
            for (int r = i + 1; r < n; ++r)
                if (std::fabs(A[r][i]) > std::fabs(A[piv][i])) piv = r;
            std::swap(A[i], A[piv]); std::swap(I[i], I[piv]);
            double a = A[i][i];
            if (std::fabs(a) < 1e-12) a = (a < 0 ? -1 : 1) * 1e-12;
            for (int j = 0; j < n; ++j) { A[i][j] /= a; I[i][j] /= a; }
            for (int r = 0; r < n; ++r) if (r != i) {
                const double f = A[r][i];
                for (int j = 0; j < n; ++j) {
                    A[r][j] -= f * A[i][j];
                    I[r][j] -= f * I[i][j];
                }
            }
        }
        return I;
    }

    int d_ = 0;
    std::vector<double> mean_;
    std::vector<std::vector<double>> invCov_;
};

// =====================================================================
//  ROC-AUC (Mann–Whitney U).  Higher score = more "normal".
//  Throws std::invalid_argument if either set is empty or any score is
//  NaN / ±inf (a NaN key would also break std::sort's strict weak order).
// =====================================================================
inline double roc_auc(const std::vector<double>& normal_scores,
                      const std::vector<double>& anomaly_scores) {
    const long long pos = (long long)normal_scores.size();
    const long long neg = (long long)anomaly_scores.size();
    if (pos == 0 || neg == 0)
        throw std::invalid_argument("roc_auc needs at least one normal and one anomaly score");
    if (!detail::all_finite(normal_scores.data(), normal_scores.size()) ||
        !detail::all_finite(anomaly_scores.data(), anomaly_scores.size()))
        throw std::invalid_argument("roc_auc scores contain NaN or inf");
    std::vector<std::pair<double, int>> all;
    all.reserve(pos + neg);
    for (double v : normal_scores)  all.push_back({v, 1});
    for (double v : anomaly_scores) all.push_back({v, 0});
    std::sort(all.begin(), all.end(),
              [](auto& a, auto& b){ return a.first < b.first; });
    std::vector<double> rank(all.size());
    size_t i = 0;
    while (i < all.size()) {
        size_t j = i;
        while (j + 1 < all.size() && all[j + 1].first == all[i].first) ++j;
        const double avg = (i + j) / 2.0 + 1.0;
        for (size_t k = i; k <= j; ++k) rank[k] = avg;
        i = j + 1;
    }
    double rsum = 0;
    for (size_t k = 0; k < all.size(); ++k)
        if (all[k].second == 1) rsum += rank[k];
    const double U = rsum - (double)pos * (pos + 1) / 2.0;
    return U / ((double)pos * (double)neg);
}

}  // namespace dmkde
