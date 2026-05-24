// dmkde.hpp — Density Matrix Kernel Density Estimation (single-header)
//
// Implementation of the algorithm described in:
//
//   Useche, D.; González, F. et al.
//   "Quantum density estimation with density matrices:
//    Application to quantum anomaly detection."
//   Phys. Rev. A 109, 032418 (2024).   arXiv:2201.10006
//
// Plus the streaming O(1)-per-sample variant from:
//
//   "INQMAD: Incremental Quantum Measurement Anomaly Detection."
//   arXiv:2210.05061 (2025).
//
// Algorithm in one paragraph:
//   Each input x ∈ R^d is embedded into R^D by Random Fourier Features
//   (RFF), which approximates a Gaussian kernel as ⟨φ(x), φ(y)⟩ ≈ k(x,y).
//   Training builds the empirical density operator
//
//       ρ = (1/N) Σ_i  φ(x_i) φ(x_i)^T     (PSD, trace 1)
//
//   For a new x, the Born-rule score
//
//       s(x) = φ(x)^T  ρ  φ(x)   ∈ [0, 1]
//
//   estimates the probability density of x under the training distribution.
//   Low s ⇒ anomalous.
//
// Streaming (INQMAD): ρ_{t+1} = (1-α) ρ_t + α · φ(x_t) φ(x_t)^T   (rank-1 EMA)
//
// This is a pure C++17 header-only library. No external dependencies.
// All linear algebra is hand-rolled and parallelizable with OpenMP.
//
// License: MIT.  See LICENSE.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace dmkde {

constexpr double PI = 3.14159265358979323846;

// =====================================================================
//  AdaptiveFourierFeatures
//  ------------------------
//  Maps R^d → R^D via stacked sin/cos projections of d-dim Gaussian
//  random frequencies. The kernel approximated is the Gaussian / RBF
//
//      k(x, y) = exp( -||x - y||^2 / (2 σ^2) )
//
//  This is the "RFF" of Rahimi & Recht (NIPS 2007) and is the embedding
//  used in the Useche 2024 / González 2022 papers for DMKDE.
// =====================================================================
class AdaptiveFourierFeatures {
public:
    AdaptiveFourierFeatures() = default;

    void fit(int input_dim, int output_dim, double sigma, uint64_t seed) {
        if (output_dim % 2 != 0)
            throw std::invalid_argument("output_dim must be even (sin+cos pairs)");
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

    // Map x ∈ R^{input_dim} → out ∈ R^{output_dim}.  out must be sized.
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

    int input_dim()  const { return input_dim_; }
    int output_dim() const { return output_dim_; }
    double sigma()   const { return sigma_; }

private:
    int input_dim_  = 0;
    int output_dim_ = 0;
    double sigma_   = 1.0;
    double scale_   = 1.0;
    std::vector<double> W_;  // (K, input_dim) row-major; K = output_dim / 2
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
    DMKDE(int input_dim, int feature_dim, double sigma, uint64_t seed = 42)
        : D_(feature_dim), rho_((size_t)feature_dim * feature_dim, 0.0) {
        if (feature_dim <= 0 || feature_dim % 2 != 0)
            throw std::invalid_argument("feature_dim must be positive and even");
        aff_.fit(input_dim, feature_dim, sigma, seed);
    }

    // Batch fit: rho = (1/N) Σ φ(x_i) φ(x_i)^T.
    // Only the upper triangle is computed and then mirrored.
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty()) throw std::invalid_argument("empty training set");
        const int d_in = aff_.input_dim();
        for (const auto& row : X)
            if ((int)row.size() != d_in)
                throw std::invalid_argument("training row dimension mismatch");

        std::fill(rho_.begin(), rho_.end(), 0.0);
        std::vector<double> phi(D_);
        for (const auto& x : X) {
            aff_.transform(x.data(), phi.data());
            // ρ += φ φ^T  (upper triangle only)
            for (int i = 0; i < D_; ++i) {
                const double phi_i = phi[i];
                double* rho_row = &rho_[(size_t)i * D_];
                for (int j = i; j < D_; ++j) {
                    rho_row[j] += phi_i * phi[j];
                }
            }
        }
        const double inv_n = 1.0 / (double)X.size();
        for (int i = 0; i < D_; ++i)
            for (int j = i; j < D_; ++j) {
                rho_[(size_t)i * D_ + j] *= inv_n;
                rho_[(size_t)j * D_ + i]  = rho_[(size_t)i * D_ + j];
            }
        n_fitted_ = (long long)X.size();
    }

    // Streaming rank-1 EMA update (INQMAD, arXiv:2210.05061).
    // ρ_{t+1} = (1 - α) ρ_t + α φ(x) φ(x)^T.
    // O(D²) per update; O(1) extra memory.
    void update(const double* x, double alpha) {
        if (alpha <= 0.0 || alpha > 1.0)
            throw std::invalid_argument("alpha must be in (0, 1]");
        std::vector<double> phi(D_);
        aff_.transform(x, phi.data());
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
    }

    // Born-rule density score: s(x) = φ(x)^T ρ φ(x).
    // Higher = more "normal"; lower = more anomalous.
    double score(const double* x) const {
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

    int feature_dim() const { return D_; }
    int input_dim()   const { return aff_.input_dim(); }
    long long n_fitted()   const { return n_fitted_; }
    long long n_streamed() const { return n_streamed_; }
    double trace() const {
        double t = 0;
        for (int i = 0; i < D_; ++i) t += rho_[(size_t)i * D_ + i];
        return t;
    }

private:
    int D_;
    std::vector<double> rho_;   // D x D row-major
    AdaptiveFourierFeatures aff_;
    long long n_fitted_   = 0;
    long long n_streamed_ = 0;
};

// =====================================================================
//  Mahalanobis baseline (for honest benchmarking).
// =====================================================================
class Mahalanobis {
public:
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty()) throw std::invalid_argument("empty training set");
        const int d = (int)X[0].size();
        d_ = d;
        mean_.assign(d, 0.0);
        for (const auto& r : X)
            for (int j = 0; j < d; ++j) mean_[j] += r[j];
        for (int j = 0; j < d; ++j) mean_[j] /= (double)X.size();

        std::vector<std::vector<double>> C(d, std::vector<double>(d, 0));
        for (const auto& r : X)
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j)
                    C[i][j] += (r[i] - mean_[i]) * (r[j] - mean_[j]);
        const double inv = 1.0 / (double)(X.size() - 1);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) C[i][j] *= inv;
        for (int i = 0; i < d; ++i) C[i][i] += 1e-6;     // ridge
        invCov_ = invert(C);
    }

    // Returns *negative* Mahalanobis distance so "higher = more normal"
    // matches the DMKDE convention — apples-to-apples for AUC.
    double score(const double* x) const {
        std::vector<double> diff(d_);
        for (int i = 0; i < d_; ++i) diff[i] = x[i] - mean_[i];
        double q = 0;
        for (int i = 0; i < d_; ++i)
            for (int j = 0; j < d_; ++j)
                q += diff[i] * invCov_[i][j] * diff[j];
        return -std::sqrt(std::max(0.0, q));
    }

private:
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
// =====================================================================
inline double roc_auc(const std::vector<double>& normal_scores,
                      const std::vector<double>& anomaly_scores) {
    const long long pos = (long long)normal_scores.size();
    const long long neg = (long long)anomaly_scores.size();
    if (pos == 0 || neg == 0) return 0.5;
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
