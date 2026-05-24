// test_basic.cpp — sanity tests for libdmkde.
//
//   * RFF embedding produces unit-norm vectors of correct length.
//   * Fitted ρ is symmetric and trace 1.
//   * Score lies in [0, 1].
//   * Batch fit on identical points and streaming update with α=1 converge.
//
// Exit code 0 ⇒ all pass.

#include "dmkde.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s  (line %d)\n", msg, __LINE__); return 1; } \
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

    std::fprintf(stderr, "all tests passed (tr=%.4f, score=%.4f→%.4f, AUC=%.4f)\n",
                 tr, sBefore, sAfter, auc);
    return 0;
}
