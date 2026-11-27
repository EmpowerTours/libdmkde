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
//   AdaptiveFourierFeatures). Since 0.2.0 the draw is portable: one seed
//   gives the same frequencies, bit for bit, with every compiler and
//   standard library (see detail::PortableNormal). 0.1.x used
//   std::normal_distribution, whose output is implementation-defined, so
//   a seed now builds a different model than it did in 0.1.x.
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
//     They do not validate values either: DMKDE::score(x) returns NaN for
//     a non-finite x or an overflowing embedding, Mahalanobis::score(x)
//     returns NaN for a non-finite x - mean and -inf for a distance
//     beyond DBL_MAX.
//   * Every other entry point (std::vector overloads and pointer+length
//     overloads) checks sizes and throws std::invalid_argument on mismatch.
//   * All fit / update paths reject NaN / ±inf (std::invalid_argument).
//   * Checked score paths also reject a finite x whose RFF embedding (or
//     Mahalanobis distance) is not finite, e.g. |x| ~ 1e308: they throw
//     std::invalid_argument and never return NaN.
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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

// Default cap on the number of RFF weights, (feature_dim/2) · input_dim. W is
// a dense array of doubles allocated and filled at construction, so without
// a cap DMKDE(input_dim = 2^31 - 1, ...) tried to fill 8 GiB per weight row.
// 2^28 weights (2 GiB) is the memory kMaxFeatureDim allows for ρ; e.g. at
// feature_dim = 256 it admits input_dim <= 2^21. Larger W needs
// allow_large = true.
constexpr size_t kMaxWeightCount = (size_t)1 << 28;

// DMKDE::from_state proves ρ + 1e-9·trace·I ≻ 0 with a full blocked
// Cholesky factorisation (O(D³/3) flops; ~0.04 s at D = 1024 and ~0.3 s at
// D = 2048 on one laptop core) up to this D. Above it, where the cubic cost
// grows to ~2.3 s at 4096 and ~20 s at 8192, it is replaced by O(D²)
// necessary conditions plus a randomised Lanczos search for a
// negative-curvature direction (37 mat-vecs, ~0.8 s at 4096), which can
// miss a small negative eigenvalue (see from_state).
constexpr int kFullPsdCheckMaxDim = 2048;

namespace detail {

inline bool all_finite(const double* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i])) return false;
    return true;
}

// n_rows * n_cols, or std::length_error if that overflows size_t.
inline size_t checked_count(size_t n_rows, size_t n_cols) {
    if (n_rows > 0 && n_cols > std::numeric_limits<size_t>::max() / n_rows)
        throw std::length_error("n_rows * n_cols overflows size_t");
    return n_rows * n_cols;
}

// True iff n == (feature_dim/2) · input_dim, decided without forming the
// product (it can exceed a 32-bit size_t: feature_dim = 8 and input_dim =
// 2^30 wrap to 0). Both dims must be > 0.
inline bool weights_size_ok(size_t n, int feature_dim, int input_dim) {
    const size_t K = (size_t)(feature_dim / 2), d = (size_t)input_dim;
    return K > 0 && n % K == 0 && n / K == d;
}

// ---- Portable random draws -------------------------------------------
// std::normal_distribution (and every other <random> distribution) is
// implementation-defined: libstdc++, libc++ and MSVC turn the same engine
// output into different numbers, so in 0.1.x one seed built a different
// model on Linux, macOS and Windows. Since 0.2.0 every draw goes through
// the samplers below, which use only the engine (std::mt19937_64, whose
// output sequence the standard specifies exactly) and IEEE-754 binary64
// operations that are correctly rounded everywhere (+, -, *, /, sqrt) plus
// the exact frexp / ldexp. No libm transcendental (log, cos, ...) is used:
// those differ by an ulp between C libraries. Each product that feeds an
// addition goes through fp_mul so that a compiler cannot contract it into
// a fused multiply-add (clang's default on arm64), which rounds once
// instead of twice. Assumes binary64 arithmetic without extended precision
// (SSE2 / NEON; not x87) and no -ffast-math.

// a * b rounded to double, never fused with a following addition.
inline double fp_mul(double a, double b) {
    volatile double r = a * b;
    return r;
}

// Uniform on [-1, 1): the top 53 bits of one engine output, k, give
// k·2^-52 - 1 exactly (every step is exact).
inline double uniform_pm1(std::mt19937_64& g) {
    return std::ldexp((double)(g() >> 11), -52) - 1.0;
}

// Natural log of a finite x > 0 (normal, not subnormal) from basic
// operations only, accurate to a few ulp and bit-identical on every
// conforming platform. x = m·2^e with m in [sqrt(1/2), sqrt(2)); then
// log(m) = 2·atanh(t), t = (m - 1)/(m + 1), |t| <= 0.1716, by its odd
// series to t^25 (truncation < 1e-19 relative).
inline double portable_log(double x) {
    int e = 0;
    double m = std::frexp(x, &e);  // exact; m in [0.5, 1)
    if (m < 0.70710678118654752440) {
        m = m * 2.0;  // exact
        --e;
    }
    const double f = m - 1.0;  // exact (Sterbenz)
    const double t = f / (2.0 + f);
    const double t2 = fp_mul(t, t);
    double p = 1.0 / 25.0;
    for (int k = 11; k >= 1; --k) p = 1.0 / (double)(2 * k + 1) + fp_mul(t2, p);
    const double two_t = 2.0 * t;  // exact
    const double lm = two_t + fp_mul(fp_mul(two_t, t2), p);
    // ln 2 split (fdlibm): e·kLn2Hi is exact for |e| < 2^11.
    constexpr double kLn2Hi = 6.93147180369123816490e-01;
    constexpr double kLn2Lo = 1.90821492927058770002e-10;
    const double de = (double)e;
    return fp_mul(de, kLn2Hi) + (fp_mul(de, kLn2Lo) + lm);
}

// Standard normal draws by the Marsaglia polar method, fully specified:
// draw u then v with uniform_pm1, retry while s = u² + v² is >= 1 or 0,
// set f = sqrt(-2·log(s) / s), and return v·f now and u·f on the next
// call (the order libstdc++ uses, so Linux models stay within rounding of
// 0.1.x). Same seed ⇒ same sequence, bit for bit, on every platform.
class PortableNormal {
public:
    explicit PortableNormal(uint64_t seed) : g_(seed) {}
    double operator()() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u, v, s;
        do {
            u = uniform_pm1(g_);
            v = uniform_pm1(g_);
            s = fp_mul(u, u) + fp_mul(v, v);
        } while (!(s < 1.0) || s == 0.0);
        const double f = std::sqrt(-2.0 * portable_log(s) / s);
        spare_ = fp_mul(u, f);
        has_spare_ = true;
        return fp_mul(v, f);
    }

private:
    std::mt19937_64 g_;
    double spare_ = 0.0;
    bool has_spare_ = false;
};

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
    // (feature_dim/2) · input_dim weights, compared without multiplying (an
    // overflowing product is above any cap).
    const size_t K = (size_t)(feature_dim / 2);
    if (!allow_large && (size_t)input_dim > kMaxWeightCount / K)
        throw std::invalid_argument(
            "input_dim=" + std::to_string(input_dim) + " with feature_dim=" +
            std::to_string(feature_dim) + " needs (feature_dim/2) * input_dim weights, above "
            "the default cap of " + std::to_string(kMaxWeightCount) + " (2 GiB); pass "
            "allow_large=true to opt in");
    detail::checked_count(K, (size_t)input_dim);  // std::length_error on overflow
}

// Shifted Cholesky test: true iff the symmetric D×D row-major matrix A
// satisfies A + shift·I ≻ 0, i.e. its smallest eigenvalue is > -shift.
// Reads only the lower triangle. O(D³/3) flops, O(D²) memory.
//
// Blocked right-looking factorisation (panel width kNB). The trailing
// update, which is almost all of the work, runs a 4×4 register-tiled
// kernel over a packed copy of the panel, so it is compute-bound instead
// of streaming the whole matrix from memory once per column as the
// unblocked left-looking loop did (D = 4096: ~35 s → ~2 s on one core).
inline bool cholesky_ok(const std::vector<double>& A0, size_t n, double shift) {
    constexpr size_t kNB = 64;
    std::vector<double> A(A0);
    std::vector<double> P;  // panel of L, packed in 4-row groups: [g][p][r]
    for (size_t k0 = 0; k0 < n; k0 += kNB) {
        const size_t kb = std::min(kNB, n - k0);
        const size_t k1 = k0 + kb;
        // 1. Factor the diagonal block (already updated by earlier panels).
        for (size_t j = 0; j < kb; ++j) {
            double* Lj = &A[(k0 + j) * n + k0];
            double s = Lj[j] + shift;
            for (size_t p = 0; p < j; ++p) s -= Lj[p] * Lj[p];
            if (!(s > 0.0) || !std::isfinite(s)) return false;
            const double ljj = std::sqrt(s);
            Lj[j] = ljj;
            for (size_t i = j + 1; i < kb; ++i) {
                double* Li = &A[(k0 + i) * n + k0];
                double t = Li[j];
                for (size_t p = 0; p < j; ++p) t -= Li[p] * Lj[p];
                Li[j] = t / ljj;
            }
        }
        if (k1 >= n) break;
        // 2. Panel below the diagonal block: L21 = A21 · L11^-T (row by row).
        for (size_t i = k1; i < n; ++i) {
            double* Li = &A[i * n + k0];
            for (size_t j = 0; j < kb; ++j) {
                const double* Lj = &A[(k0 + j) * n + k0];
                double t = Li[j];
                for (size_t p = 0; p < j; ++p) t -= Li[p] * Lj[p];
                Li[j] = t / Lj[j];
            }
            if (!detail::all_finite(Li, kb)) return false;
        }
        // 3. Trailing update A22 -= L21 · L21^T (lower triangle only).
        const size_t m = n - k1;
        const size_t groups = (m + 3) / 4;
        P.assign(groups * kb * 4, 0.0);
        for (size_t r = 0; r < m; ++r) {
            const double* Lr = &A[(k1 + r) * n + k0];
            double* dst = &P[(r / 4) * kb * 4 + (r % 4)];
            for (size_t p = 0; p < kb; ++p) dst[p * 4] = Lr[p];
        }
        for (size_t gi = 0; gi < groups; ++gi) {
            const double* Pi = &P[gi * kb * 4];
            for (size_t gj = 0; gj <= gi; ++gj) {
                const double* Pj = &P[gj * kb * 4];
                double acc[4][4] = {{0}};
                for (size_t p = 0; p < kb; ++p) {
                    const double* a = Pi + p * 4;
                    const double* b = Pj + p * 4;
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c) acc[r][c] += a[r] * b[c];
                }
                for (size_t r = 0; r < 4; ++r) {
                    const size_t i = gi * 4 + r;
                    if (i >= m) break;
                    double* Ai = &A[(k1 + i) * n + k1];
                    for (size_t c = 0; c < 4; ++c) {
                        const size_t j = gj * 4 + c;
                        if (j > i) break;
                        Ai[j] -= acc[r][c];
                    }
                }
            }
        }
    }
    return true;
}

// Symmetric eigen-decomposition of a small dense k×k matrix T (row-major)
// by cyclic Jacobi rotations. On return T's diagonal holds the eigenvalues
// and V (k×k, row-major) the eigenvectors as columns.
inline void jacobi_eigen(std::vector<double>& T, std::vector<double>& V, size_t k) {
    V.assign(k * k, 0.0);
    for (size_t i = 0; i < k; ++i) V[i * k + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, tot = 0;
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j) {
                tot += T[i * k + j] * T[i * k + j];
                if (i != j) off += T[i * k + j] * T[i * k + j];
            }
        if (off <= 1e-30 * tot || off == 0.0) return;
        for (size_t p = 0; p < k; ++p)
            for (size_t q = p + 1; q < k; ++q) {
                const double apq = T[p * k + q];
                if (apq == 0.0) continue;
                const double theta = (T[q * k + q] - T[p * k + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (size_t r = 0; r < k; ++r) {  // columns p, q
                    const double tp = T[r * k + p], tq = T[r * k + q];
                    T[r * k + p] = c * tp - s * tq;
                    T[r * k + q] = s * tp + c * tq;
                }
                for (size_t r = 0; r < k; ++r) {  // rows p, q
                    const double tp = T[p * k + r], tq = T[q * k + r];
                    T[p * k + r] = c * tp - s * tq;
                    T[q * k + r] = s * tp + c * tq;
                }
                for (size_t r = 0; r < k; ++r) {
                    const double vp = V[r * k + p], vq = V[r * k + q];
                    V[r * k + p] = c * vp - s * vq;
                    V[r * k + q] = s * vp + c * vq;
                }
            }
    }
}

// A fresh, unpredictable seed for each PSD probe: std::random_device mixed
// (splitmix64) with a process-wide call counter and the clock, so neither
// two calls nor two processes share probe vectors even where random_device
// is deterministic. A fixed seed would make the probe vectors public and let
// a crafted state hide a negative eigenvalue orthogonal to all of them.
inline uint64_t fresh_probe_seed() {
    static std::atomic<uint64_t> counter{0};
    uint64_t x = counter.fetch_add(1, std::memory_order_relaxed) * 0x9E3779B97F4A7C15ull;
    x ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    try {
        std::random_device rd;
        x ^= ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    } catch (...) {
        // No entropy source: the counter and the clock still differ per call.
    }
    x += 0x9E3779B97F4A7C15ull;  // splitmix64 finaliser
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Randomised search for a certificate that the symmetric n×n row-major A
// is NOT positive semidefinite to tolerance `shift`: a vector v with
// vᵀAv < -shift·‖v‖². Returns false iff such a v is found, so it never
// rejects a PSD matrix (every rejection is an explicitly computed Rayleigh
// quotient). It can miss a small negative eigenvalue: it is a probe, not
// a proof of PSD-ness. Cost: (n_probes + 2·steps + 1) dense mat-vecs.
//
// Probes: n_probes Gaussian vectors, then `steps` Lanczos iterations with
// full reorthogonalisation from a further Gaussian start. Lanczos finds the
// extreme ends of the spectrum first, so a negative eigenvalue that is not
// tiny relative to the spread of the spectrum shows up as the smallest
// Ritz value; its Ritz vector's Rayleigh quotient is then recomputed
// exactly against A.
inline bool psd_probe_ok(const std::vector<double>& A, size_t n, double shift,
                         int n_probes, int steps, uint64_t seed) {
    PortableNormal g(seed);
    std::vector<double> v(n), w(n);
    auto matvec = [&](const double* x, double* y) {
        for (size_t i = 0; i < n; ++i) {
            const double* Ai = &A[i * n];
            double s = 0;
            for (size_t j = 0; j < n; ++j) s += Ai[j] * x[j];
            y[i] = s;
        }
    };
    auto dot = [&](const double* x, const double* y) {
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
        return s;
    };
    auto rayleigh_bad = [&](const double* x) {
        matvec(x, w.data());
        const double xx = dot(x, x);
        return xx > 0 && dot(x, w.data()) < -shift * xx;
    };
    for (int t = 0; t < n_probes; ++t) {
        for (auto& e : v) e = g();
        if (rayleigh_bad(v.data())) return false;
    }
    const size_t k = std::min<size_t>((size_t)std::max(steps, 1), n);
    std::vector<double> Q(k * n), alpha(k, 0.0), beta(k, 0.0);
    for (auto& e : v) e = g();
    double nv = std::sqrt(dot(v.data(), v.data()));
    for (size_t i = 0; i < n; ++i) Q[i] = v[i] / nv;
    size_t used = 0;
    for (size_t it = 0; it < k; ++it) {
        const double* q = &Q[it * n];
        matvec(q, w.data());
        alpha[it] = dot(q, w.data());
        used = it + 1;
        if (alpha[it] < -shift) return false;  // q is unit: a direct certificate
        if (it + 1 == k) break;
        // Full reorthogonalisation (twice) against all previous vectors.
        for (int pass = 0; pass < 2; ++pass)
            for (size_t j = 0; j <= it; ++j) {
                const double* qj = &Q[j * n];
                const double c = dot(qj, w.data());
                for (size_t i = 0; i < n; ++i) w[i] -= c * qj[i];
            }
        const double b = std::sqrt(dot(w.data(), w.data()));
        if (!(b > 1e-300) || !std::isfinite(b)) break;  // invariant subspace
        beta[it] = b;
        double* qn = &Q[(it + 1) * n];
        for (size_t i = 0; i < n; ++i) qn[i] = w[i] / b;
    }
    std::vector<double> T(used * used, 0.0), V;
    for (size_t i = 0; i < used; ++i) {
        T[i * used + i] = alpha[i];
        if (i + 1 < used) T[i * used + i + 1] = T[(i + 1) * used + i] = beta[i];
    }
    jacobi_eigen(T, V, used);
    size_t imin = 0;
    for (size_t i = 1; i < used; ++i)
        if (T[i * used + i] < T[imin * used + imin]) imin = i;
    // Ritz vector x = Q^T y for the smallest Ritz value; check it exactly.
    std::fill(v.begin(), v.end(), 0.0);
    for (size_t j = 0; j < used; ++j) {
        const double y = V[j * used + imin];
        const double* qj = &Q[j * n];
        for (size_t i = 0; i < n; ++i) v[i] += y * qj[i];
    }
    return !rayleigh_bad(v.data());
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

        // ω ~ N(0, σ^-2 I): W[k][j] = z · (1/σ) for the standard normal
        // draws z of detail::PortableNormal(seed), taken in row-major order.
        // This is the same on every platform (0.1.x used
        // std::normal_distribution, whose output is implementation-defined).
        detail::PortableNormal gauss(seed);
        const double inv_sigma = 1.0 / sigma;
        W_.assign(detail::checked_count((size_t)K, (size_t)input_dim), 0.0);
        for (size_t i = 0; i < W_.size(); ++i) W_[i] = detail::fp_mul(gauss(), inv_sigma);

        scale_ = std::sqrt(2.0 / (double)output_dim);
    }

    // Install explicit frequencies W (K × input_dim row-major, K = output_dim/2).
    // Used to restore a saved model bit-for-bit, including models built by
    // 0.1.x (whose std::normal_distribution draw differed per platform).
    void set_weights(int input_dim, int output_dim, double sigma,
                     const std::vector<double>& W) {
        if (input_dim <= 0)
            throw std::invalid_argument("input_dim must be > 0");
        if (output_dim <= 0 || output_dim % 2 != 0)
            throw std::invalid_argument("output_dim must be positive and even (cos+sin pairs)");
        detail::check_sigma(sigma);
        if (!detail::weights_size_ok(W.size(), output_dim, input_dim))
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

    // Size-checked transform; rejects non-finite x and a finite x whose
    // embedding is not finite (input too large), like the checked score.
    void transform(const double* x, size_t n, double* out, size_t out_n) const {
        if (output_dim_ == 0) throw std::logic_error("AdaptiveFourierFeatures used before fit()");
        if (n != (size_t)input_dim_)
            throw std::invalid_argument("x has " + std::to_string(n) + " features, expected " +
                                        std::to_string(input_dim_));
        if (out_n != (size_t)output_dim_)
            throw std::invalid_argument("output buffer has wrong size");
        if (!detail::all_finite(x, n)) throw std::invalid_argument("x contains NaN or inf");
        transform(x, out);
        if (!detail::all_finite(out, out_n))
            throw std::invalid_argument("RFF embedding of x is not finite (input too large)");
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
        if (!detail::all_finite(X, detail::checked_count(n_rows, n_cols)))
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

    // Atomic batch update over a row-major (n_rows × n_cols) buffer: the
    // same arithmetic as calling update() on each row in order, but every
    // row (finite input AND finite RFF embedding) is validated before any
    // row is applied, so on a throw ρ and the counters are unchanged.
    void update_batch(const double* X, size_t n_rows, size_t n_cols, double alpha) {
        require_fitted("update_batch");
        if (!(alpha > 0.0 && alpha <= 1.0))
            throw std::invalid_argument("alpha must be in (0, 1]");
        check_cols(n_cols);
        if (!detail::all_finite(X, detail::checked_count(n_rows, n_cols)))
            throw std::invalid_argument("X contains NaN or inf");
        std::vector<double> phi(D_);
        for (size_t r = 0; r < n_rows; ++r) {
            aff_.transform(X + r * n_cols, phi.data());
            if (!detail::all_finite(phi.data(), phi.size()))
                throw std::invalid_argument("RFF embedding of row " + std::to_string(r) +
                                            " is not finite (input too large)");
        }
        for (size_t r = 0; r < n_rows; ++r) update(X + r * n_cols, alpha);
    }

    // Born-rule density score: s(x) = φ(x)^T ρ φ(x), clamped to [0, 1].
    // Higher = more "normal"; lower = more anomalous.
    // UNCHECKED width: x must hold input_dim() doubles. A non-finite x, or
    // one whose RFF embedding overflows, yields NaN. Throws
    // std::logic_error before fit().
    double score(const double* x) const { return score_row(x, false); }

    // Size-checked score; also rejects non-finite x and a finite x whose
    // RFF embedding is not finite (input too large).
    double score(const double* x, size_t n) const {
        check_cols(n);
        if (!detail::all_finite(x, n))
            throw std::invalid_argument("x contains NaN or inf");
        return score_row(x, true);
    }
    double score(const std::vector<double>& x) const { return score(x.data(), x.size()); }

    // Size-checked batch score over a row-major (n_rows × n_cols) buffer.
    // out must hold n_rows doubles. Rejects non-finite input and rows whose
    // RFF embedding is not finite.
    void score_batch(const double* X, size_t n_rows, size_t n_cols, double* out) const {
        require_fitted("score_batch");
        check_cols(n_cols);
        if (!detail::all_finite(X, detail::checked_count(n_rows, n_cols)))
            throw std::invalid_argument("X contains NaN or inf");
        for (size_t i = 0; i < n_rows; ++i) out[i] = score_row(X + i * n_cols, true);
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

    // Rebuild a model from a saved state. Validates shapes, finiteness of
    // W and ρ, symmetry, trace 1, the counters and positive
    // semidefiniteness of ρ to tolerance 1e-9·trace; throws
    // std::invalid_argument on any inconsistency.
    //
    // PSD check. A state is accepted iff ρ + 1e-9·trace·I is positive
    // definite, i.e. every eigenvalue of ρ is > -1e-9·trace (a matrix whose
    // smallest eigenvalue is exactly -1e-9·trace sits on the rejection
    // boundary and may go either way by rounding):
    //   * D <= kFullPsdCheckMaxDim (2048): proved by a shifted, blocked
    //     Cholesky factorisation, O(D³).
    //   * larger D: O(D²) necessary conditions (non-negative diagonal,
    //     every 2×2 principal minor ρ_ij² <= (ρ_ii + t)(ρ_jj + t)) and a
    //     randomised search (4 Gaussian probes and 32 Lanczos steps) for a
    //     v with vᵀρv < -t·‖v‖². Every rejection is backed by such an
    //     explicitly computed v, so a valid ρ is never rejected, but a
    //     negative eigenvalue that is small relative to the spread of ρ's
    //     spectrum can be missed. The probe vectors are drawn from a fresh
    //     seed on every call (fresh_probe_seed), so a crafted state cannot
    //     hide a negative direction orthogonal to known probes; still, this
    //     path is PROBABILISTIC and NOT adversarially robust: a negative
    //     eigenvalue tiny relative to the spectrum can pass. Only load
    //     D > 2048 states from trusted sources. The probe costs ~0.8 s at
    //     D = 4096 where a Cholesky would take ~2.3 s (8192: a few s vs ~20 s).
    static DMKDE from_state(const DMKDEState& s, bool allow_large = false) {
        // Check the stored array sizes against the claimed dimensions BEFORE
        // constructing: the constructor draws a fresh (feature_dim/2) x
        // input_dim random W and a D x D rho, so a state claiming a huge
        // input_dim with a tiny W would otherwise allocate (and fill)
        // gigabytes to terabytes before set_weights rejects it. With these
        // checks the work done is bounded by the data the state carries.
        // The sizes are compared by division, never by a product that could
        // wrap on a 32-bit size_t (feature_dim = 8, input_dim = 2^30 gave 0
        // and let an empty W through to set_weights' own check only).
        // check_dims (run first) rejects non-positive dims, the weight cap
        // and an overflowing feature_dim^2.
        detail::check_dims(s.input_dim, s.feature_dim, allow_large);
        if (!detail::weights_size_ok(s.W.size(), s.feature_dim, s.input_dim))
            throw std::invalid_argument("W must have (feature_dim/2) * input_dim entries");
        const size_t D = (size_t)s.feature_dim;
        if (s.rho.size() % D != 0 || s.rho.size() / D != D)
            throw std::invalid_argument("rho must have feature_dim^2 entries");
        DMKDE m(s.input_dim, s.feature_dim, s.sigma, s.seed, allow_large);
        m.aff_.set_weights(s.input_dim, s.feature_dim, s.sigma, s.W);
        if (!detail::all_finite(s.rho.data(), s.rho.size()))
            throw std::invalid_argument("rho contains NaN or inf");
        if (s.n_fitted < 0 || s.n_streamed < 0)
            throw std::invalid_argument("counters must be non-negative");
        if (s.fitted) {
            if (s.n_fitted < 1)
                throw std::invalid_argument("a fitted state needs n_fitted >= 1");
            const std::vector<double>& r = s.rho;
            double tr = 0;
            for (size_t i = 0; i < D; ++i) tr += r[i * D + i];
            if (std::fabs(tr - 1.0) > 1e-6)
                throw std::invalid_argument("rho must have trace 1");
            const double shift = 1e-9 * tr;
            // Symmetry, non-negative diagonal and every 2×2 principal minor
            // (ρ_ij² ≤ (ρ_ii + shift)(ρ_jj + shift)): O(D²) necessary
            // conditions for ρ + shift·I ⪰ 0. Walked in 64×64 tiles so the
            // transposed reads stay in cache.
            for (size_t i = 0; i < D; ++i)
                if (!(r[i * D + i] >= -shift))
                    throw std::invalid_argument(
                        "rho is not positive semidefinite (negative diagonal entry)");
            constexpr size_t kT = 64;
            for (size_t i0 = 0; i0 < D; i0 += kT)
                for (size_t j0 = i0; j0 < D; j0 += kT)
                    for (size_t i = i0; i < std::min(i0 + kT, D); ++i)
                        for (size_t j = std::max(j0, i + 1); j < std::min(j0 + kT, D); ++j) {
                            const double a = r[i * D + j], b = r[j * D + i];
                            if (std::fabs(a - b) > 1e-12 * (1.0 + std::fabs(a)))
                                throw std::invalid_argument("rho is not symmetric");
                            if (a * a > (r[i * D + i] + shift) * (r[j * D + j] + shift))
                                throw std::invalid_argument(
                                    "rho is not positive semidefinite (a 2x2 principal "
                                    "minor is negative)");
                        }
            if (D <= (size_t)kFullPsdCheckMaxDim) {
                if (!detail::cholesky_ok(r, D, shift))
                    throw std::invalid_argument(
                        "rho is not positive semidefinite (smallest eigenvalue < -1e-9 * trace)");
            } else if (!detail::psd_probe_ok(r, D, shift, 4, 32, detail::fresh_probe_seed())) {
                throw std::invalid_argument(
                    "rho is not positive semidefinite (found v with v^T rho v < -1e-9 * "
                    "trace * |v|^2)");
            }
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

    // Born-rule score of one row. With check = true a non-finite RFF
    // embedding throws instead of yielding NaN. The arithmetic is the same
    // either way (the check only reads φ).
    double score_row(const double* x, bool check) const {
        require_fitted("score");
        std::vector<double> phi(D_), rphi(D_, 0.0);
        aff_.transform(x, phi.data());
        if (check && !detail::all_finite(phi.data(), phi.size()))
            throw std::invalid_argument("RFF embedding of x is not finite (input too large)");
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
        // In exact arithmetic s is in [0, 1] (ρ is PSD with trace 1 and
        // ‖φ‖ = 1); rounding can put it an ulp or so outside (e.g. 1 + 2^-52
        // for a point equal to every training row), so clamp. NaN stays NaN.
        if (s > 1.0) return 1.0;
        if (s < 0.0) return 0.0;
        return s;
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
//  The covariance gets a per-feature relative ridge, C + 1e-6 · diag(C),
//  so the score is invariant (up to rounding) to rescaling each
//  non-constant feature independently, i.e. to the units of every column.
//  Constant features use an absolute scale instead, P_ii = 1e6 (see
//  fit_impl), independent of the other columns.
// =====================================================================
class Mahalanobis {
public:
    // Per-feature relative ridge: C + kRidge · diag(C).
    static constexpr double kRidge = 1e-6;

    // Needs N >= 2 finite rows of equal width d >= 1.
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.size() < 2)
            throw std::invalid_argument("Mahalanobis needs at least 2 training rows");
        const size_t d = X[0].size();
        if (d == 0) throw std::invalid_argument("training rows have 0 features");
        if (d > (size_t)std::numeric_limits<int>::max())
            throw std::invalid_argument("training rows have too many features");
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
        if (n_cols > (size_t)std::numeric_limits<int>::max())
            throw std::invalid_argument("training rows have too many features");
        if (!detail::all_finite(X, detail::checked_count(n_rows, n_cols)))
            throw std::invalid_argument("training data contains NaN or inf");
        fit_impl(n_rows, n_cols, [&](size_t i) { return X + i * n_cols; });
    }

    // Returns *negative* Mahalanobis distance so "higher = more normal"
    // matches the DMKDE convention — apples-to-apples for AUC.
    // UNCHECKED width: x must hold input_dim() doubles. Non-finite x
    // yields NaN (never the "most normal" score). Throws before fit().
    // If the quadratic form overflows for a finite x (|x| ~ 1e308), it is
    // recomputed on diff / max|diff| and rescaled. The result is then a
    // very large finite distance, or -inf when the distance itself exceeds
    // DBL_MAX (e.g. x - mean finite but near DBL_MAX in a direction of
    // small variance); NaN if x - mean itself is not finite. Use the
    // checked overloads, which throw instead of returning -inf or NaN.
    // For ordinary input the arithmetic is unchanged.
    double score(const double* x) const {
        if (d_ == 0) throw std::logic_error("Mahalanobis::score called before fit()");
        std::vector<double> diff(d_);
        for (int i = 0; i < d_; ++i) diff[i] = x[i] - mean_[i];
        double q = quad_form(diff);
        if (std::isfinite(q)) return -std::sqrt(std::max(0.0, q));
        double m = 0;
        for (int i = 0; i < d_; ++i) {
            if (!std::isfinite(diff[i])) return std::numeric_limits<double>::quiet_NaN();
            m = std::max(m, std::fabs(diff[i]));
        }
        for (int i = 0; i < d_; ++i) diff[i] /= m;
        q = quad_form(diff);
        if (!std::isfinite(q)) return std::numeric_limits<double>::quiet_NaN();
        return -(m * std::sqrt(std::max(0.0, q)));
    }

    // Size-checked score; rejects non-finite x, and a finite x whose
    // distance is not finite (x - mean overflows).
    double score(const double* x, size_t n) const {
        if (d_ == 0) throw std::logic_error("Mahalanobis::score called before fit()");
        if (n != (size_t)d_)
            throw std::invalid_argument("x has " + std::to_string(n) + " features, expected " +
                                        std::to_string(d_));
        if (!detail::all_finite(x, n)) throw std::invalid_argument("x contains NaN or inf");
        return checked_score(x);
    }
    double score(const std::vector<double>& x) const { return score(x.data(), x.size()); }

    void score_batch(const double* X, size_t n_rows, size_t n_cols, double* out) const {
        if (d_ == 0) throw std::logic_error("Mahalanobis::score_batch called before fit()");
        if (n_cols != (size_t)d_)
            throw std::invalid_argument("X has " + std::to_string(n_cols) + " features, expected " +
                                        std::to_string(d_));
        if (!detail::all_finite(X, detail::checked_count(n_rows, n_cols)))
            throw std::invalid_argument("X contains NaN or inf");
        for (size_t i = 0; i < n_rows; ++i) out[i] = checked_score(X + i * n_cols);
    }

    int input_dim() const { return d_; }
    bool fitted() const { return d_ > 0; }

    // Serialisation: mean (d) and inverse covariance (d × d, row-major).
    const std::vector<double>& mean() const { return mean_; }
    std::vector<double> inv_cov() const {
        std::vector<double> out(detail::checked_count((size_t)d_, (size_t)d_));
        for (int i = 0; i < d_; ++i)
            for (int j = 0; j < d_; ++j) out[(size_t)i * d_ + j] = invCov_[i][j];
        return out;
    }
    // Requires finite mean and a finite, symmetric (to 1e-8 relative to its
    // largest entry), positive-definite inv_cov (Cholesky, O(d³)).
    void set_state(const std::vector<double>& mean, const std::vector<double>& inv_cov) {
        const size_t d = mean.size();
        if (d == 0) throw std::invalid_argument("mean must be non-empty");
        if (d > (size_t)std::numeric_limits<int>::max())
            throw std::invalid_argument("mean has too many entries");
        if (inv_cov.size() != detail::checked_count(d, d))
            throw std::invalid_argument("inv_cov must have d*d entries");
        if (!detail::all_finite(mean.data(), d) || !detail::all_finite(inv_cov.data(), d * d))
            throw std::invalid_argument("Mahalanobis state contains NaN or inf");
        double amax = 0;
        for (double v : inv_cov) amax = std::max(amax, std::fabs(v));
        for (size_t i = 0; i < d; ++i)
            for (size_t j = i + 1; j < d; ++j)
                if (std::fabs(inv_cov[i * d + j] - inv_cov[j * d + i]) > 1e-8 * amax)
                    throw std::invalid_argument("inv_cov is not symmetric");
        if (!detail::cholesky_ok(inv_cov, d, 0.0))
            throw std::invalid_argument("inv_cov is not positive definite");
        std::vector<std::vector<double>> P(d, std::vector<double>(d));
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) P[i][j] = inv_cov[i * d + j];
        mean_ = mean;
        invCov_ = std::move(P);
        d_ = (int)d;
    }

private:
    double quad_form(const std::vector<double>& diff) const {
        double q = 0;
        for (int i = 0; i < d_; ++i)
            for (int j = 0; j < d_; ++j)
                q += diff[i] * invCov_[i][j] * diff[j];
        return q;
    }

    double checked_score(const double* x) const {
        const double s = score(x);
        if (!std::isfinite(s))
            throw std::invalid_argument(
                "Mahalanobis distance of x is not finite (input too large)");
        return s;
    }

    template <class RowFn>
    void fit_impl(size_t n_rows, size_t n_cols, RowFn row_at) {
        const int d = (int)n_cols;
        // Work on each column divided by a power of two a_i close to its
        // largest magnitude (so the division is exact), z_i = x_i / a_i with
        // |z_i| < 2.
        // Means and covariances of z can neither overflow nor underflow, so a
        // feature's variance is measured correctly whatever its units
        // (1e-300 or 1e300): there is no squaring of raw values.
        std::vector<double> scale(d, 0.0);
        // A column whose min equals its max is exactly constant. Its mean is
        // that value exactly; a summed mean can drift by ~n·eps relative
        // (0.1 is not exactly summable) and would give the column a tiny
        // spurious variance.
        std::vector<double> cmin(d), cmax(d);
        for (size_t r = 0; r < n_rows; ++r) {
            const double* x = row_at(r);
            for (int j = 0; j < d; ++j) {
                scale[j] = std::max(scale[j], std::fabs(x[j]));
                if (r == 0 || x[j] < cmin[j]) cmin[j] = x[j];
                if (r == 0 || x[j] > cmax[j]) cmax[j] = x[j];
            }
        }
        std::vector<char> exact_const(d, 0);
        for (int j = 0; j < d; ++j) exact_const[j] = (cmin[j] == cmax[j]) ? 1 : 0;
        for (int j = 0; j < d; ++j) {
            int e = 1;
            // frexp: max|x_j| = m·2^e with m in [0.5, 1), so a_j = 2^(e-1)
            // satisfies max|x_j| / a_j in [1, 2) and stays finite even for
            // max|x_j| near DBL_MAX (2^e would overflow there).
            if (scale[j] > 0) std::frexp(scale[j], &e);
            scale[j] = scale[j] > 0 ? std::ldexp(1.0, e - 1) : 1.0;
        }
        std::vector<double> zmean(d, 0.0);
        for (size_t r = 0; r < n_rows; ++r) {
            const double* x = row_at(r);
            for (int j = 0; j < d; ++j) zmean[j] += x[j] / scale[j];
        }
        for (int j = 0; j < d; ++j) zmean[j] /= (double)n_rows;
        // One refinement pass: add the mean of the residuals, which removes
        // the rounding error of the summed mean to first order.
        {
            std::vector<double> corr(d, 0.0);
            for (size_t r = 0; r < n_rows; ++r) {
                const double* x = row_at(r);
                for (int j = 0; j < d; ++j) corr[j] += x[j] / scale[j] - zmean[j];
            }
            for (int j = 0; j < d; ++j) zmean[j] += corr[j] / (double)n_rows;
        }
        for (int j = 0; j < d; ++j)
            if (exact_const[j]) zmean[j] = cmin[j] / scale[j];

        std::vector<std::vector<double>> C(d, std::vector<double>(d, 0));
        std::vector<double> z(d);
        for (size_t r = 0; r < n_rows; ++r) {
            const double* x = row_at(r);
            for (int j = 0; j < d; ++j) z[j] = x[j] / scale[j] - zmean[j];
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) C[i][j] += z[i] * z[j];
        }
        const double inv = 1.0 / (double)(n_rows - 1);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) C[i][j] *= inv;

        // Per-feature relative ridge: P = (Σ + kRidge·diag(Σ))^-1, Σ the
        // covariance of x. With v_i = Σ_ii it is computed as
        // V^-1/2 (R + kRidge·I)^-1 V^-1/2, R = V^-1/2 Σ V^-1/2 the
        // correlation matrix (the same for x and z). Rescaling feature i by
        // b_i maps Σ to BΣB, V to B²V and leaves R unchanged, so the
        // distance (x-μ)ᵀP(x-μ) is invariant to per-feature units (up to
        // rounding).
        //
        // Constant features. A feature with min == max has no scale to be
        // relative to. It gets an ABSOLUTE unit scale, v_i = 1, no
        // correlations, and the ridge kRidge, so P_ii = 1/kRidge = 1e6 (as
        // the fixed 1e-6·I ridge of 0.1.0 gave): a deviation δ from the
        // constant adds 1e6·δ² to the squared distance (distance 1000·|δ|
        // alone), whatever the units of the other features. Its score is
        // therefore NOT invariant to rescaling that feature itself. If every
        // feature is constant the fit is rejected.
        //
        // v_i = a_i²·var(z_i) is never formed: P_ij = Pn_ij·t_i·t_j with
        // t_i = 1/(a_i·sqrt(var(z_i))) = 1/std(x_i). If t_i² overflows (the
        // feature's spread is below ~1e-154) or underflows (above ~1e154),
        // the inverse covariance cannot be stored in double precision and the
        // fit is rejected with a message naming the feature.
        std::vector<char> constant(d, 0);
        int nvar = 0;
        for (int i = 0; i < d; ++i) {
            // Constant means min == max, nothing looser: the mean is exact for
            // such a column and refined otherwise, so any column with two
            // distinct values has a real, measurable variance (even a 1-ulp
            // spread of a huge value) and is scaled by it. A rounding
            // tolerance here classed real spreads under ~64 ulp as constant
            // and gave them a meaningless absolute scale.
            if (exact_const[i] || !(C[i][i] > 0.0)) constant[i] = 1;
            else ++nvar;
        }
        if (nvar == 0)
            throw std::invalid_argument(
                "training data has zero (or non-finite) variance in every feature");
        std::vector<double> zsd(d, 1.0), t(d, 1.0);
        for (int i = 0; i < d; ++i) {
            if (constant[i]) continue;
            zsd[i] = std::sqrt(C[i][i]);
            t[i] = 1.0 / (scale[i] * zsd[i]);
            const double t2 = t[i] * t[i];
            if (!std::isfinite(t2))
                throw std::invalid_argument(
                    "Mahalanobis fit underflowed: feature " + std::to_string(i) +
                    " has a standard deviation too small (< ~1e-154) for its inverse "
                    "variance to be represented; rescale it");
            if (!(t2 >= std::numeric_limits<double>::min()))
                throw std::invalid_argument(
                    "Mahalanobis fit overflowed: feature " + std::to_string(i) +
                    " has a standard deviation too large (> ~1e154) for its inverse "
                    "variance to be represented; rescale it");
        }
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                C[i][j] = (constant[i] || constant[j]) ? 0.0 : C[i][j] / (zsd[i] * zsd[j]);
        for (int i = 0; i < d; ++i) C[i][i] = (constant[i] ? 0.0 : 1.0) + kRidge;
        auto P = invert(C);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) P[i][j] = (P[i][j] * t[i]) * t[j];
        std::vector<double> mean(d);
        for (int i = 0; i < d; ++i) mean[i] = zmean[i] * scale[i];
        for (int i = 0; i < d; ++i) {
            if (!std::isfinite(mean[i]))
                throw std::invalid_argument(
                    "Mahalanobis fit overflowed (mean of feature " + std::to_string(i) +
                    " not representable; rescale the features)");
            for (int j = 0; j < d; ++j) {
                if (std::isfinite(P[i][j])) continue;
                // Each feature's own 1/std² is representable (checked above),
                // so the overflow comes from a large entry of the inverse
                // correlation matrix, i.e. strongly correlated features.
                // Name feature i's most strongly correlated partner.
                int k = -1;
                for (int q = 0; q < d; ++q)
                    if (q != i && !constant[q] && !constant[i] &&
                        (k < 0 || std::fabs(C[i][q]) > std::fabs(C[i][k])))
                        k = q;
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.9g", k < 0 ? 0.0 : C[i][k]);
                const int a = k < 0 ? i : std::min(i, k), b = k < 0 ? j : std::max(i, k);
                throw std::invalid_argument(
                    "Mahalanobis fit overflowed: the inverse covariance of features " +
                    std::to_string(a) + " and " + std::to_string(b) +
                    " is not representable: they are strongly correlated (correlation " + buf +
                    ", which inflates the inverse variance by about 1/(1 - r^2)) and their "
                    "standard deviations are small (1/std^2 already near the top of the double "
                    "range); rescale them (e.g. standardize) or drop one of the correlated "
                    "features");
            }
        }

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
