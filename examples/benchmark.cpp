// benchmark.cpp — Reproducible benchmark for libdmkde.
//
// Generates three synthetic anomaly-detection scenarios and reports
// ROC-AUC for both DMKDE (Born-rule density-matrix scoring) and a
// Mahalanobis baseline:
//
//   1. Gaussian normal       — single tight Gaussian (the linear
//                              baseline's mathematically optimal case).
//   2. Bimodal manifold      — two well-separated clusters with
//                              anomalies hiding in the gap between
//                              them (where the empirical mean sits).
//   3. Ring + outliers       — normal data lives on a 2-D ring,
//                              anomalies cluster at the ring's centre.
//
// Build:  see Makefile.
// Run:    ./benchmark

#include "dmkde.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using std::vector;

enum class Scenario { Gaussian, BimodalManifold, Ring };

static void gen_data(Scenario sc, int d, int nTrain, int nTestNorm, int nTestAnom,
                     uint64_t seed,
                     vector<vector<double>>& train,
                     vector<vector<double>>& testN,
                     vector<vector<double>>& testA) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N01(0, 1);

    auto rand_d = [&](double sigma, const vector<double>& mu){
        vector<double> x(d);
        for (int j = 0; j < d; ++j) x[j] = mu[j] + sigma * N01(rng);
        return x;
    };

    if (sc == Scenario::Gaussian) {
        vector<double> mu(d, 0.0);
        for (int j = 0; j < d; ++j) mu[j] = 0.5 * std::cos(j * 0.7);
        train.reserve(nTrain);
        for (int i = 0; i < nTrain;     ++i) train.push_back(rand_d(0.6, mu));
        testN.reserve(nTestNorm);
        for (int i = 0; i < nTestNorm;  ++i) testN.push_back(rand_d(0.6, mu));
        testA.reserve(nTestAnom);
        std::uniform_real_distribution<double> U(-4.0, 4.0);
        while ((int)testA.size() < nTestAnom) {
            vector<double> x(d);
            double dist2 = 0;
            for (int j = 0; j < d; ++j) { x[j] = U(rng); dist2 += (x[j]-mu[j])*(x[j]-mu[j]); }
            if (dist2 > (double)d * 2.5) testA.push_back(std::move(x));
        }
        return;
    }

    if (sc == Scenario::BimodalManifold) {
        vector<double> mu1(d), mu2(d);
        for (int j = 0; j < d; ++j) {
            mu1[j] = (j % 2 == 0) ? -1.8 :  1.2;
            mu2[j] = (j % 2 == 0) ?  1.8 : -1.2;
        }
        auto sample_bimodal = [&](int n, vector<vector<double>>& out){
            std::bernoulli_distribution coin(0.5);
            for (int i = 0; i < n; ++i)
                out.push_back(rand_d(0.35, coin(rng) ? mu1 : mu2));
        };
        sample_bimodal(nTrain,    train);
        sample_bimodal(nTestNorm, testN);
        // anomalies near origin — exactly where the empirical mean sits
        for (int i = 0; i < nTestAnom; ++i) {
            vector<double> x(d);
            for (int j = 0; j < d; ++j) x[j] = 0.4 * N01(rng);
            testA.push_back(std::move(x));
        }
        return;
    }

    // Ring: normals on a (d-dim hyper-) ring of radius 3, anomalies at centre.
    auto on_ring = [&](){
        vector<double> x(d);
        for (int j = 0; j < d; ++j) x[j] = N01(rng);
        double n = 0; for (double v : x) n += v * v;
        n = std::sqrt(n);
        if (n < 1e-9) n = 1;
        std::normal_distribution<double> radial(3.0, 0.18);
        const double r = std::max(0.5, radial(rng));
        for (int j = 0; j < d; ++j) x[j] = r * x[j] / n;
        return x;
    };
    for (int i = 0; i < nTrain;    ++i) train.push_back(on_ring());
    for (int i = 0; i < nTestNorm; ++i) testN.push_back(on_ring());
    for (int i = 0; i < nTestAnom; ++i) {
        vector<double> x(d);
        for (int j = 0; j < d; ++j) x[j] = 0.4 * N01(rng);  // hub
        testA.push_back(std::move(x));
    }
}

static void run_scenario(const char* name, Scenario sc,
                         int d, int nTrain, int nTestN, int nTestA,
                         int featureDim, double sigma, uint64_t seed) {
    std::printf("\n======================================================\n");
    std::printf("  %s\n", name);
    std::printf("======================================================\n");
    std::printf("  d=%d  D=%d (RFF dim)  σ=%g  N_train=%d  N_test=%d+%d\n",
                d, featureDim, sigma, nTrain, nTestN, nTestA);

    vector<vector<double>> train, testN, testA;
    gen_data(sc, d, nTrain, nTestN, nTestA, seed, train, testN, testA);

    // ---- Mahalanobis baseline ----
    dmkde::Mahalanobis maha;
    auto t0 = std::chrono::high_resolution_clock::now();
    maha.fit(train);
    auto t1 = std::chrono::high_resolution_clock::now();
    vector<double> mN(testN.size()), mA(testA.size());
    for (size_t i = 0; i < testN.size(); ++i) mN[i] = maha.score(testN[i].data());
    for (size_t i = 0; i < testA.size(); ++i) mA[i] = maha.score(testA[i].data());
    auto t2 = std::chrono::high_resolution_clock::now();
    const double mAuc = dmkde::roc_auc(mN, mA);
    const int mFit = (int)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const int mInf = (int)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // ---- DMKDE ----
    dmkde::DMKDE model(d, featureDim, sigma, seed + 1);
    auto u0 = std::chrono::high_resolution_clock::now();
    model.fit(train);
    auto u1 = std::chrono::high_resolution_clock::now();
    vector<double> qN(testN.size()), qA(testA.size());
    for (size_t i = 0; i < testN.size(); ++i) qN[i] = model.score(testN[i].data());
    for (size_t i = 0; i < testA.size(); ++i) qA[i] = model.score(testA[i].data());
    auto u2 = std::chrono::high_resolution_clock::now();
    const double qAuc = dmkde::roc_auc(qN, qA);
    const int qFit = (int)std::chrono::duration_cast<std::chrono::microseconds>(u1 - u0).count();
    const int qInf = (int)std::chrono::duration_cast<std::chrono::microseconds>(u2 - u1).count();

    std::printf("\n  Mahalanobis   AUC = %.4f   fit %5d µs   infer %6d µs\n",
                mAuc, mFit, mInf);
    std::printf("  DMKDE         AUC = %.4f   fit %5d µs   infer %6d µs  trace(ρ)=%.4f\n",
                qAuc, qFit, qInf, model.trace());
    std::printf("  →  %s by %.4f AUC\n",
                qAuc > mAuc ? "DMKDE wins" : "Mahalanobis wins",
                std::fabs(qAuc - mAuc));
}

// ---- Quick streaming demo (INQMAD variant) ----
static void demo_streaming(int d, int featureDim, double sigma, uint64_t seed) {
    std::printf("\n======================================================\n");
    std::printf("  STREAMING demo (INQMAD rank-1 EMA update)\n");
    std::printf("======================================================\n");

    vector<vector<double>> warmup, testN, testA;
    gen_data(Scenario::BimodalManifold, d, 200, 200, 200, seed + 9, warmup, testN, testA);

    dmkde::DMKDE model(d, featureDim, sigma, seed + 5);
    model.fit(warmup);

    // Feed each test-normal point through update() (drift simulation),
    // then score the remaining anomalies — model should still flag them.
    const double alpha = 1.0 / 200.0;  // ~200-sample memory horizon
    auto u0 = std::chrono::high_resolution_clock::now();
    for (const auto& x : testN) model.update(x.data(), alpha);
    auto u1 = std::chrono::high_resolution_clock::now();

    vector<double> sN(testN.size()), sA(testA.size());
    for (size_t i = 0; i < testN.size(); ++i) sN[i] = model.score(testN[i].data());
    for (size_t i = 0; i < testA.size(); ++i) sA[i] = model.score(testA[i].data());
    const double auc = dmkde::roc_auc(sN, sA);
    const int upd_us = (int)std::chrono::duration_cast<std::chrono::microseconds>(u1 - u0).count();

    std::printf("  α = %g   updates = %lld   per-update cost = %.2f µs\n",
                alpha, model.n_streamed(),
                (double)upd_us / (double)testN.size());
    std::printf("  Post-streaming AUC = %.4f   trace(ρ) = %.4f\n", auc, model.trace());
}

int main(int argc, char** argv) {
    int d           = argc > 1 ? std::atoi(argv[1]) : 4;
    int featureDim  = argc > 2 ? std::atoi(argv[2]) : 256;
    double sigma    = argc > 3 ? std::atof(argv[3]) : 1.5;
    int nTrain      = argc > 4 ? std::atoi(argv[4]) : 400;
    int nTest       = argc > 5 ? std::atoi(argv[5]) : 300;
    uint64_t seed   = argc > 6 ? (uint64_t)std::atoll(argv[6]) : 42;

    std::printf("libdmkde benchmark — production C++ implementation of\n");
    std::printf("  Useche et al., Phys. Rev. A 109, 032418 (2024)\n");
    std::printf("  arXiv:2201.10006   +   INQMAD streaming variant arXiv:2210.05061\n");

    run_scenario("Scenario 1: Gaussian normal (Mahalanobis-optimal case)",
                 Scenario::Gaussian, d, nTrain, nTest, nTest,
                 featureDim, sigma, seed);

    run_scenario("Scenario 2: Bimodal manifold (anomalies in the gap)",
                 Scenario::BimodalManifold, d, nTrain, nTest, nTest,
                 featureDim, sigma, seed);

    run_scenario("Scenario 3: Ring manifold (anomalies in the hub)",
                 Scenario::Ring, d, nTrain, nTest, nTest,
                 featureDim, sigma, seed);

    demo_streaming(d, featureDim, sigma, seed);
    return 0;
}
