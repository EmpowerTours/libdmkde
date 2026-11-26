// test_basic.cpp — sanity tests for libdmkde.
//
//   * RFF embedding produces unit-norm vectors of correct length.
//   * Fitted ρ is symmetric and trace 1.
//   * Score lies in [0, 1].
//   * Batch fit on identical points and streaming update with α=1 converge.
//   * Invalid input (0.1.1): bad parameters, wrong widths, NaN / inf,
//     use before fit, Mahalanobis N < 2, roc_auc NaN / empty, state
//     round-trip and state validation. Run under -fsanitize=address,undefined.
//
// Exit code 0 ⇒ all pass.

#include "dmkde.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s  (line %d)\n", msg, __LINE__); return 1; } \
    } while (0)

// Evaluate expr; pass iff it throws exactly Exc (or a subclass).
#define ASSERT_THROWS(Exc, expr, msg) do { \
    bool caught_ = false; \
    try { (void)(expr); } catch (const Exc&) { caught_ = true; } catch (...) {} \
    ASSERT(caught_, msg); \
    } while (0)

static double l2(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x * x; return std::sqrt(s);
}

int main() {
    using namespace dmkde;
    const int d = 4, D = 64;

    // ---- Test 1: AFF dims + norm ----
    AdaptiveFourierFeatures aff;
    aff.fit(d, D, 1.5, 7);
    std::vector<double> x = {0.3, -0.8, 1.1, 0.0};
    std::vector<double> phi(D);
    aff.transform(x.data(), phi.data());
    const double n = l2(phi);
    ASSERT(n > 0.5 && n < 1.5, "AFF norm in plausible range");

    // ---- Test 2: trace ≈ 1 after batch fit on unit-norm embeddings ----
    DMKDE model(d, D, 1.5, 7);
    std::vector<std::vector<double>> X = {
        {0.1, 0.2, -0.3, 0.5},
        {0.4, 0.0, -0.1, 0.2},
        {0.0, 0.1,  0.2, 0.3},
        {-0.5, 0.3, 0.4, -0.1},
    };
    model.fit(X);
    const double tr = model.trace();
    ASSERT(tr > 0.4 && tr < 1.6, "trace(ρ) within plausible bounds");

    // ---- Test 3: score in (0, 1) on training points (not exactly 1) ----
    double s = model.score(X[0].data());
    ASSERT(s > 0.0 && s < 1.0, "score in (0,1)");

    // ---- Test 4: streaming update changes the model ----
    DMKDE m2(d, D, 1.5, 7);
    m2.fit(X);
    const double sBefore = m2.score(X[0].data());
    std::vector<double> y = {2.5, -2.5, 2.5, -2.5};  // far point
    for (int i = 0; i < 50; ++i) m2.update(y.data(), 0.05);
    const double sAfter = m2.score(X[0].data());
    ASSERT(sAfter < sBefore, "score on old point drops after streaming on a new point");

    // ---- Test 5: ROC-AUC degenerate cases ----
    std::vector<double> a = {1, 2, 3}, b = {0, 0, 0};
    const double auc = roc_auc(a, b);
    ASSERT(auc > 0.99 && auc <= 1.0, "AUC = 1 when normals strictly > anomalies");

    // ---- Test 6 (0.1.1): invalid parameters ----
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double Inf = std::numeric_limits<double>::infinity();
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, D, 0.0), "sigma = 0 rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, D, -1.5), "sigma < 0 rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, D, NaN), "sigma NaN rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, D, Inf), "sigma inf rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(0, D, 1.0), "input_dim 0 rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(-1, D, 1.0), "input_dim < 0 rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, 0, 1.0), "feature_dim 0 rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, -64, 1.0), "feature_dim < 0 rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, 63, 1.0), "odd feature_dim rejected");
    ASSERT_THROWS(std::invalid_argument, DMKDE(d, kMaxFeatureDim + 2, 1.0),
                  "feature_dim above cap rejected without allow_large");
    {
        AdaptiveFourierFeatures bad;
        ASSERT_THROWS(std::invalid_argument, bad.fit(d, D, 0.0, 1), "AFF sigma 0 rejected");
        ASSERT_THROWS(std::invalid_argument, bad.fit(0, D, 1.0, 1), "AFF input_dim 0 rejected");
    }

    // ---- Test 7: use before fit ----
    {
        DMKDE u(d, D, 1.5, 7);
        ASSERT(!u.fitted(), "new model is unfitted");
        ASSERT_THROWS(std::logic_error, u.update(x.data(), 0.1), "update before fit throws");
        ASSERT_THROWS(std::logic_error, u.score(x.data()), "score before fit throws");
        ASSERT_THROWS(std::logic_error, u.score(x), "checked score before fit throws");
    }

    // ---- Test 8: width and finiteness checks ----
    {
        DMKDE w(d, D, 1.5, 7);
        w.fit(X);
        std::vector<double> short_row = {0.1, 0.2};
        std::vector<double> long_row  = {0.1, 0.2, 0.3, 0.4, 0.5};
        std::vector<double> nan_row   = {0.1, NaN, 0.3, 0.4};
        std::vector<double> inf_row   = {0.1, Inf, 0.3, 0.4};
        ASSERT_THROWS(std::invalid_argument, w.score(short_row), "short row rejected (score)");
        ASSERT_THROWS(std::invalid_argument, w.score(long_row),  "long row rejected (score)");
        ASSERT_THROWS(std::invalid_argument, w.score(short_row.data(), short_row.size()),
                      "pointer+length score checks width");
        ASSERT_THROWS(std::invalid_argument, w.score(nan_row), "NaN row rejected (score)");
        ASSERT_THROWS(std::invalid_argument, w.update(short_row, 0.1), "short row rejected (update)");
        ASSERT_THROWS(std::invalid_argument, w.transform(short_row), "short row rejected (transform)");
        std::vector<double> out(4);
        ASSERT_THROWS(std::invalid_argument,
                      w.score_batch(short_row.data(), 1, short_row.size(), out.data()),
                      "score_batch checks width");
        ASSERT_THROWS(std::invalid_argument,
                      w.score_batch(nan_row.data(), 1, nan_row.size(), out.data()),
                      "score_batch rejects NaN");

        const double before = w.score(X[1]);
        ASSERT_THROWS(std::invalid_argument, w.update(nan_row, 0.1), "NaN update rejected");
        ASSERT_THROWS(std::invalid_argument, w.update(inf_row, 0.1), "inf update rejected");
        ASSERT_THROWS(std::invalid_argument, w.update(X[0], NaN), "NaN alpha rejected");
        ASSERT_THROWS(std::invalid_argument, w.update(X[0], 0.0), "alpha 0 rejected");
        ASSERT_THROWS(std::invalid_argument, w.update(X[0], 1.5), "alpha > 1 rejected");
        ASSERT(w.score(X[1]) == before, "rejected updates leave rho unchanged");

        std::vector<std::vector<double>> Xnan = X;
        Xnan[2][1] = NaN;
        ASSERT_THROWS(std::invalid_argument, w.fit(Xnan), "NaN training row rejected");
        ASSERT(w.score(X[1]) == before, "rejected fit leaves rho unchanged");
        std::vector<std::vector<double>> ragged = X;
        ragged[1].pop_back();
        ASSERT_THROWS(std::invalid_argument, w.fit(ragged), "ragged training rows rejected");
        ASSERT_THROWS(std::invalid_argument, w.fit(std::vector<std::vector<double>>{}),
                      "empty training set rejected");
        std::vector<double> flat = {0.1, 0.2, 0.3};
        ASSERT_THROWS(std::invalid_argument, w.fit(flat.data(), 1, flat.size()),
                      "flat fit checks width");

        // Batch overload equals per-row score; flat fit equals vector fit.
        std::vector<double> rows;
        for (const auto& r : X) rows.insert(rows.end(), r.begin(), r.end());
        std::vector<double> batch(X.size());
        w.score_batch(rows.data(), X.size(), (size_t)d, batch.data());
        for (size_t i = 0; i < X.size(); ++i)
            ASSERT(batch[i] == w.score(X[i].data()), "score_batch == score");
        DMKDE flatm(d, D, 1.5, 7);
        flatm.fit(rows.data(), X.size(), (size_t)d);
        ASSERT(flatm.rho() == w.rho(), "flat fit == vector fit");

        // fit resets the streaming counter.
        w.update(X[0], 0.1);
        ASSERT(w.n_streamed() == 1, "update counted");
        w.fit(X);
        ASSERT(w.n_streamed() == 0, "fit resets n_streamed");

        // ---- Test 9: state round-trip and validation ----
        w.update(X[3], 0.2);
        DMKDEState st = w.state();
        DMKDE r = DMKDE::from_state(st);
        ASSERT(r.rho() == w.rho() && r.weights() == w.weights(), "state round-trip exact");
        ASSERT(r.score(X[0]) == w.score(X[0]), "restored model scores identically");
        ASSERT(r.n_streamed() == 1 && r.n_fitted() == (long long)X.size(), "counters restored");
        DMKDEState bad = st; bad.W.pop_back();
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "short W rejected");
        // A state claiming a huge input_dim with its small W must be rejected before the
        // constructor allocates (feature_dim/2) x input_dim random weights (fuzz finding).
        bad = st; bad.input_dim = 2147483647;
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "huge input_dim rejected before allocating");
        bad = st; bad.rho.pop_back();
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "short rho rejected");
        bad = st; bad.rho[1] += 0.5;
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "asymmetric rho rejected");
        bad = st; bad.rho[0] = NaN;
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "NaN rho rejected");
        bad = st; bad.n_fitted = -1;
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "negative counter rejected");
        bad = st; bad.sigma = 0.0;
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "bad sigma in state rejected");
        bad = st; for (auto& v : bad.rho) v = 0.0;
        bad.rho[0] = 2.0; bad.rho[(size_t)D + 1] = -1.0;          // diag(2, -1, 0, ...): trace 1
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "non-PSD rho rejected");
        bad = st; bad.W[0] = std::numeric_limits<double>::infinity();
        ASSERT_THROWS(std::invalid_argument, DMKDE::from_state(bad), "inf W rejected");

        // ---- Test 9b: huge finite input, atomic update_batch (0.1.1 rev3) ----
        std::vector<double> huge_row(d, 0.0); huge_row[0] = 1.7e308;
        ASSERT_THROWS(std::invalid_argument, w.score(huge_row), "huge finite row rejected (score)");
        std::vector<double> hb(huge_row);
        ASSERT_THROWS(std::invalid_argument, w.score_batch(hb.data(), 1, hb.size(), &hb[0]),
                      "huge finite row rejected (score_batch)");
        ASSERT(std::isnan(w.score(huge_row.data())), "raw score of huge row -> NaN");
        std::vector<double> ub;
        for (int k = 0; k < 3; ++k) ub.insert(ub.end(), X[k].begin(), X[k].end());
        ub.insert(ub.end(), huge_row.begin(), huge_row.end());
        const auto rho_before = w.rho();
        const long long ns_before = w.n_streamed();
        ASSERT_THROWS(std::invalid_argument, w.update_batch(ub.data(), 4, (size_t)d, 0.3),
                      "update_batch rejects a batch with a huge row");
        ASSERT(w.rho() == rho_before && w.n_streamed() == ns_before, "rejected batch leaves rho intact");
        DMKDE w2 = DMKDE::from_state(w.state());
        w.update_batch(ub.data(), 3, (size_t)d, 0.3);
        for (int k = 0; k < 3; ++k) w2.update(X[k], 0.3);
        ASSERT(w.rho() == w2.rho() && w.n_streamed() == w2.n_streamed(), "update_batch == per-row update");

        // Non-finite input is named as such (not reported as an RFF overflow):
        // the up-front finite checks in score(x, n) and update_batch.
        std::vector<double> nan_q(d, 0.0); nan_q[1] = NaN;
        auto msg_of = [](auto&& fn) -> std::string {
            try { fn(); } catch (const std::invalid_argument& e) { return e.what(); }
            return "";
        };
        ASSERT(msg_of([&]{ (void)w.score(nan_q); }).find("NaN or inf") != std::string::npos,
               "score(x, n) names NaN input");
        std::vector<double> nb;
        for (int k = 0; k < 2; ++k) nb.insert(nb.end(), X[k].begin(), X[k].end());
        nb.insert(nb.end(), nan_q.begin(), nan_q.end());
        ASSERT(msg_of([&]{ w.update_batch(nb.data(), 3, (size_t)d, 0.3); }).find("NaN or inf") != std::string::npos,
               "update_batch names NaN input");
    }

    // ---- Test 10: Mahalanobis ----
    {
        Mahalanobis mh;
        ASSERT_THROWS(std::logic_error, mh.score(x.data()), "Mahalanobis score before fit throws");
        ASSERT_THROWS(std::invalid_argument, mh.fit({X[0]}), "Mahalanobis N=1 rejected");
        std::vector<std::vector<double>> zero(5, std::vector<double>{});
        ASSERT_THROWS(std::invalid_argument, mh.fit(zero), "Mahalanobis 0 features rejected");
        std::vector<std::vector<double>> Xnan = X;
        Xnan[0][0] = NaN;
        ASSERT_THROWS(std::invalid_argument, mh.fit(Xnan), "Mahalanobis NaN training rejected");
        std::vector<std::vector<double>> constant(5, std::vector<double>(d, 1.0));
        ASSERT_THROWS(std::invalid_argument, mh.fit(constant), "Mahalanobis zero variance rejected");
        mh.fit(X);
        std::vector<double> short_row = {0.1};
        ASSERT_THROWS(std::invalid_argument, mh.score(short_row), "Mahalanobis width checked");
        std::vector<double> nan_row = {NaN, 0, 0, 0};
        ASSERT_THROWS(std::invalid_argument, mh.score(nan_row), "Mahalanobis NaN query rejected");
        ASSERT(std::isnan(mh.score(nan_row.data())), "raw Mahalanobis NaN query -> NaN, not 'most normal'");
        // Scale invariance of the relative ridge.
        std::vector<std::vector<double>> Xs = X;
        for (auto& r : Xs) for (auto& v : r) v *= 1e-4;
        Mahalanobis ms; ms.fit(Xs);
        std::vector<double> q = {0.3, -0.2, 0.1, 0.4}, qs = q;
        for (auto& v : qs) v *= 1e-4;
        ASSERT(std::fabs(ms.score(qs) - mh.score(q)) < 1e-6 * std::fabs(mh.score(q)),
               "Mahalanobis is scale-invariant");
        // Per-feature rescaling (1e-4 .. 1e4) leaves the distance unchanged.
        const double fac[4] = {1e-4, 1e4, 1.0, 1e2};
        std::vector<std::vector<double>> Xf = X;
        for (auto& r : Xf) for (int j = 0; j < d; ++j) r[j] *= fac[j];
        Mahalanobis mf; mf.fit(Xf);
        std::vector<double> qf = q;
        for (int j = 0; j < d; ++j) qf[j] *= fac[j];
        ASSERT(std::fabs(mf.score(qf) - mh.score(q)) < 1e-9 * std::fabs(mh.score(q)),
               "Mahalanobis is invariant to per-feature rescaling");
        // Overflowing quadratic form for a finite query: finite, very negative.
        std::vector<double> hq = {1e300, -1e300, 0, 0};
        const double hs = mh.score(hq);
        ASSERT(std::isfinite(hs) && hs < -1e299, "Mahalanobis huge query -> finite, most anomalous");
        // A distance beyond DBL_MAX is rejected by the checked path, NaN never returned.
        std::vector<double> hq2 = {1.7e308, -1.7e308, 1.7e308, 0};
        ASSERT_THROWS(std::invalid_argument, mh.score(hq2), "Mahalanobis overflowing distance rejected");
        // State import requires finite, symmetric, positive-definite inv_cov.
        std::vector<double> mean = mh.mean(), P = mh.inv_cov(), negI((size_t)d * d, 0.0);
        for (int i = 0; i < d; ++i) negI[(size_t)i * d + i] = -1.0;
        Mahalanobis rs;
        ASSERT_THROWS(std::invalid_argument, rs.set_state(mean, negI), "negative-definite inv_cov rejected");
        std::vector<double> asymP = P; asymP[1] += 1.0;
        ASSERT_THROWS(std::invalid_argument, rs.set_state(mean, asymP), "asymmetric inv_cov rejected");
        rs.set_state(mean, P);
        ASSERT(rs.score(q) == mh.score(q), "Mahalanobis state round-trip exact");
    }

    // ---- Test 10b: PSD checks (blocked Cholesky and the large-D probe) ----
    {
        // A = Q diag(l) Q^T for a Householder Q; smallest eigenvalue lmin.
        auto make = [](size_t n, double lmin) {
            std::vector<double> u(n), A(n * n, 0.0);
            double nu = 0;
            for (size_t i = 0; i < n; ++i) { u[i] = std::sin(1.0 + 3.7 * (double)i); nu += u[i] * u[i]; }
            for (auto& x : u) x /= std::sqrt(nu);
            std::vector<double> l(n, 1.0 / (double)n);
            l[n - 1] = lmin;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    // (I - 2uu^T) diag(l) (I - 2uu^T)
                    double s = 0;
                    for (size_t k = 0; k < n; ++k) {
                        const double qik = (i == k) - 2 * u[i] * u[k];
                        const double qjk = (j == k) - 2 * u[j] * u[k];
                        s += qik * l[k] * qjk;
                    }
                    A[i * n + j] = s;
                }
            return A;
        };
        const size_t sizes[] = {1, 3, 63, 64, 65, 130};
        for (size_t n : sizes) {
            ASSERT(detail::cholesky_ok(make(n, 0.0), n, 1e-9), "cholesky accepts PSD");
            ASSERT(detail::cholesky_ok(make(n, -5e-10), n, 1e-9), "cholesky accepts within tolerance");
            ASSERT(!detail::cholesky_ok(make(n, -3e-9), n, 1e-9), "cholesky rejects lambda < -tol");
            ASSERT(detail::psd_probe_ok(make(n, 0.0), n, 1e-9, 4, 32, 1), "probe never rejects PSD");
            if (n > 1)
                ASSERT(!detail::psd_probe_ok(make(n, -0.5), n, 1e-9, 4, 32, 1),
                       "probe finds a large negative eigenvalue");
        }
    }

    // ---- Test 11: roc_auc input validation ----
    {
        std::vector<double> nan_s = {1, NaN, 3}, empty;
        ASSERT_THROWS(std::invalid_argument, roc_auc(nan_s, b), "roc_auc rejects NaN");
        ASSERT_THROWS(std::invalid_argument, roc_auc(a, empty), "roc_auc rejects empty");
    }

    std::fprintf(stderr, "all tests passed (tr=%.4f, score=%.4f→%.4f, AUC=%.4f)\n",
                 tr, sBefore, sAfter, auc);
    return 0;
}
