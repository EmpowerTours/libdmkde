// bindings.cpp — pybind11 bindings exposing libdmkde to Python with numpy.
//
// Built as the `dmkde._core` extension module; the Python-facing API
// (sklearn-style fit / score_samples / decision_function / predict)
// lives in `dmkde/__init__.py` and wraps these primitives.

#include "dmkde.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;
using std::vector;

using NPArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Convert a 2-D numpy array (N, d) into a vector<vector<double>> the
// library expects. Materialises a copy — cheap for typical N×D.
static vector<vector<double>> as_rows(NPArrayD X) {
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be 2-D (got " + std::to_string(X.ndim()) + "-D)");
    const ssize_t N = X.shape(0);
    const ssize_t d = X.shape(1);
    auto buf = X.unchecked<2>();
    vector<vector<double>> rows((size_t)N, vector<double>((size_t)d));
    for (ssize_t i = 0; i < N; ++i)
        for (ssize_t j = 0; j < d; ++j)
            rows[(size_t)i][(size_t)j] = buf(i, j);
    return rows;
}

// Validate a 1-D row of expected length.
static const double* as_row(NPArrayD x, int expected_dim) {
    if (x.ndim() != 1)
        throw std::invalid_argument("x must be 1-D");
    if ((int)x.shape(0) != expected_dim)
        throw std::invalid_argument("x has wrong feature count");
    return x.data();
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "libdmkde — density-matrix kernel density estimation (C++ core).\n"
              "Algorithm by Useche et al., Phys. Rev. A 109, 032418 (2024).";

    py::class_<dmkde::DMKDE>(m, "DMKDE",
        "Density-matrix kernel density estimator.\n\n"
        "Parameters\n"
        "----------\n"
        "input_dim : int   — dimensionality of input features.\n"
        "feature_dim : int — RFF embedding dimension D (must be even).\n"
        "sigma : float     — Gaussian-kernel bandwidth.\n"
        "seed : int        — RNG seed for the random Fourier projections.\n")
        .def(py::init<int, int, double, uint64_t>(),
             py::arg("input_dim"),
             py::arg("feature_dim") = 256,
             py::arg("sigma") = 1.0,
             py::arg("seed") = 42)
        .def("fit",
             [](dmkde::DMKDE& self, NPArrayD X){
                 self.fit(as_rows(X));
             },
             py::arg("X"),
             "Batch-fit ρ = (1/N) Σ φ(xᵢ) φ(xᵢ)ᵀ from the training rows.")
        .def("update",
             [](dmkde::DMKDE& self, NPArrayD x, double alpha){
                 self.update(as_row(x, self.input_dim()), alpha);
             },
             py::arg("x"), py::arg("alpha"),
             "Streaming rank-1 EMA update: ρ ← (1-α)ρ + α φ(x) φ(x)ᵀ.")
        .def("score",
             [](const dmkde::DMKDE& self, NPArrayD x){
                 return self.score(x.data());
             },
             py::arg("x"),
             "Born-rule density score for one row; higher = more 'normal'.")
        .def("score_batch",
             [](const dmkde::DMKDE& self, NPArrayD X){
                 if (X.ndim() != 2)
                     throw std::invalid_argument("X must be 2-D");
                 const ssize_t N = X.shape(0);
                 NPArrayD out(N);
                 auto in = X.unchecked<2>();
                 auto out_mut = out.mutable_unchecked<1>();
                 vector<double> row(X.shape(1));
                 for (ssize_t i = 0; i < N; ++i) {
                     for (ssize_t j = 0; j < X.shape(1); ++j) row[j] = in(i, j);
                     out_mut(i) = self.score(row.data());
                 }
                 return out;
             },
             py::arg("X"),
             "Vectorised score over rows; returns a 1-D numpy array.")
        .def_property_readonly("feature_dim", &dmkde::DMKDE::feature_dim)
        .def_property_readonly("input_dim",   &dmkde::DMKDE::input_dim)
        .def_property_readonly("trace",       &dmkde::DMKDE::trace)
        .def_property_readonly("n_fitted",    &dmkde::DMKDE::n_fitted)
        .def_property_readonly("n_streamed",  &dmkde::DMKDE::n_streamed);

    py::class_<dmkde::Mahalanobis>(m, "Mahalanobis",
        "Mahalanobis distance baseline (negative distance = higher = more 'normal').")
        .def(py::init<>())
        .def("fit",
             [](dmkde::Mahalanobis& self, NPArrayD X){ self.fit(as_rows(X)); },
             py::arg("X"))
        .def("score",
             [](const dmkde::Mahalanobis& self, NPArrayD x){ return self.score(x.data()); },
             py::arg("x"))
        .def("score_batch",
             [](const dmkde::Mahalanobis& self, NPArrayD X){
                 if (X.ndim() != 2)
                     throw std::invalid_argument("X must be 2-D");
                 const ssize_t N = X.shape(0);
                 NPArrayD out(N);
                 auto in = X.unchecked<2>();
                 auto out_mut = out.mutable_unchecked<1>();
                 vector<double> row(X.shape(1));
                 for (ssize_t i = 0; i < N; ++i) {
                     for (ssize_t j = 0; j < X.shape(1); ++j) row[j] = in(i, j);
                     out_mut(i) = self.score(row.data());
                 }
                 return out;
             },
             py::arg("X"));

    m.def("roc_auc",
          [](py::array_t<double> normal, py::array_t<double> anomaly){
              vector<double> n(normal.data(), normal.data() + normal.size());
              vector<double> a(anomaly.data(), anomaly.data() + anomaly.size());
              return dmkde::roc_auc(n, a);
          },
          py::arg("normal_scores"), py::arg("anomaly_scores"),
          "Mann-Whitney ROC-AUC; higher 'normal' score is treated as positive.");
}
