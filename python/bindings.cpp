// bindings.cpp — pybind11 bindings exposing libdmkde to Python with numpy.
//
// Built as the `dmkde._core` extension module; the Python-facing API
// (sklearn-style fit / score_samples / decision_function / predict)
// lives in `dmkde/__init__.py` and wraps these primitives.
//
// Safety rules enforced here (every entry point):
//   * inputs are converted to C-contiguous float64 (c_style | forcecast),
//     so strided / non-float views are copied, never read as raw memory;
//   * array rank and width are checked against the model before any
//     pointer reaches the C++ core; (N, 0) arrays are rejected;
//   * NaN / inf are rejected (ValueError);
//   * the heavy loops run with the GIL released, under a per-object
//     reader/writer lock (concurrent scoring, exclusive fit/update).

#include "dmkde.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using std::vector;

using NPArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

// DMKDE plus the lock that makes it safe to call from several Python
// threads once the GIL is released.
struct PyDMKDE {
    dmkde::DMKDE m;
    mutable std::shared_mutex mu;

    PyDMKDE(int input_dim, int feature_dim, double sigma, uint64_t seed, bool allow_large)
        : m(input_dim, feature_dim, sigma, seed, allow_large) {}
    explicit PyDMKDE(dmkde::DMKDE&& mm) : m(std::move(mm)) {}
};

struct PyMahalanobis {
    dmkde::Mahalanobis m;
    mutable std::shared_mutex mu;
};

// Release the GIL, then take the object's lock (in that order, so a thread
// waiting for the lock never holds the GIL).
template <class Obj, class F>
auto read_locked(const Obj& o, F&& f) {
    py::gil_scoped_release nogil;
    std::shared_lock<std::shared_mutex> lk(o.mu);
    return f();
}

template <class Obj, class F>
auto write_locked(Obj& o, F&& f) {
    py::gil_scoped_release nogil;
    std::unique_lock<std::shared_mutex> lk(o.mu);
    return f();
}

void check_2d(const NPArrayD& X, const char* name) {
    if (X.ndim() != 2)
        throw std::invalid_argument(std::string(name) + " must be 2-D (n_samples, n_features), got " +
                                    std::to_string(X.ndim()) + "-D");
}

void check_width(const NPArrayD& X, int expected, const char* name) {
    check_2d(X, name);
    if (X.shape(1) != (py::ssize_t)expected)
        throw std::invalid_argument(std::string(name) + " has " + std::to_string(X.shape(1)) +
                                    " features, but the model expects " +
                                    std::to_string(expected));
}

void check_row(const NPArrayD& x, int expected, const char* name) {
    if (x.ndim() != 1)
        throw std::invalid_argument(std::string(name) + " must be 1-D, got " +
                                    std::to_string(x.ndim()) + "-D");
    if (x.shape(0) != (py::ssize_t)expected)
        throw std::invalid_argument(std::string(name) + " has " + std::to_string(x.shape(0)) +
                                    " features, but the model expects " +
                                    std::to_string(expected));
}

NPArrayD matrix_copy(const vector<double>& flat, py::ssize_t rows, py::ssize_t cols) {
    NPArrayD out({rows, cols});
    std::copy(flat.begin(), flat.end(), out.mutable_data());
    return out;
}

py::dict state_to_dict(const dmkde::DMKDEState& s, bool allow_large) {
    py::dict d;
    d["format"]      = 1;
    d["input_dim"]   = s.input_dim;
    d["feature_dim"] = s.feature_dim;
    d["sigma"]       = s.sigma;
    d["seed"]        = s.seed;
    d["allow_large"] = allow_large;
    d["W"]           = matrix_copy(s.W, s.feature_dim / 2, s.input_dim);
    d["rho"]         = matrix_copy(s.rho, s.feature_dim, s.feature_dim);
    d["n_fitted"]    = s.n_fitted;
    d["n_streamed"]  = s.n_streamed;
    d["fitted"]      = s.fitted;
    return d;
}

dmkde::DMKDEState dict_to_state(const py::dict& d, bool& allow_large) {
    static const char* keys[] = {"format", "input_dim", "feature_dim", "sigma", "seed",
                                 "W", "rho", "n_fitted", "n_streamed", "fitted"};
    for (const char* k : keys)
        if (!d.contains(k)) throw std::invalid_argument(std::string("state is missing '") + k + "'");
    if (d["format"].cast<int>() != 1)
        throw std::invalid_argument("unsupported state format");
    dmkde::DMKDEState s;
    s.input_dim   = d["input_dim"].cast<int>();
    s.feature_dim = d["feature_dim"].cast<int>();
    s.sigma       = d["sigma"].cast<double>();
    s.seed        = d["seed"].cast<uint64_t>();
    s.n_fitted    = d["n_fitted"].cast<long long>();
    s.n_streamed  = d["n_streamed"].cast<long long>();
    s.fitted      = d["fitted"].cast<bool>();
    allow_large   = d.contains("allow_large") ? d["allow_large"].cast<bool>() : false;
    // Validate dims before trusting them for shape checks.
    dmkde::detail::check_dims(s.input_dim, s.feature_dim, allow_large);
    NPArrayD W   = NPArrayD::ensure(d["W"]);
    NPArrayD rho = NPArrayD::ensure(d["rho"]);
    if (!W || !rho) throw std::invalid_argument("W and rho must be float arrays");
    if (W.ndim() != 2 || W.shape(0) != s.feature_dim / 2 || W.shape(1) != s.input_dim)
        throw std::invalid_argument("W must have shape (feature_dim/2, input_dim)");
    if (rho.ndim() != 2 || rho.shape(0) != s.feature_dim || rho.shape(1) != s.feature_dim)
        throw std::invalid_argument("rho must have shape (feature_dim, feature_dim)");
    s.W.assign(W.data(), W.data() + W.size());
    s.rho.assign(rho.data(), rho.data() + rho.size());
    return s;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "libdmkde — density-matrix kernel density estimation (C++ core).\n"
              "Algorithm by Useche, Bustos-Brinez, Gallego-Mejia and González,\n"
              "Phys. Rev. A 109, 032418 (2024), arXiv:2201.10006.";

    m.attr("MAX_FEATURE_DIM") = dmkde::kMaxFeatureDim;

    py::class_<PyDMKDE>(m, "DMKDE",
        "Density-matrix kernel density estimator.\n\n"
        "Parameters\n"
        "----------\n"
        "input_dim : int    — dimensionality of input features (> 0).\n"
        "feature_dim : int  — RFF embedding dimension D (positive, even,\n"
        "                     <= MAX_FEATURE_DIM unless allow_large=True).\n"
        "sigma : float      — Gaussian-kernel bandwidth (finite, > 0). The score\n"
        "                     averages the squared kernel, so the effective\n"
        "                     bandwidth is sigma / sqrt(2).\n"
        "seed : int         — RNG seed for the random Fourier projections.\n"
        "allow_large : bool — permit feature_dim > MAX_FEATURE_DIM.\n")
        .def(py::init<int, int, double, uint64_t, bool>(),
             py::arg("input_dim"),
             py::arg("feature_dim") = 256,
             py::arg("sigma") = 1.0,
             py::arg("seed") = 42,
             py::arg("allow_large") = false)
        .def("fit",
             [](PyDMKDE& self, NPArrayD X){
                 check_width(X, self.m.input_dim(), "X");
                 const double* p = X.data();
                 const size_t n = (size_t)X.shape(0), d = (size_t)X.shape(1);
                 write_locked(self, [&]{ self.m.fit(p, n, d); return 0; });
             },
             py::arg("X"),
             "Batch-fit ρ = (1/N) Σ φ(xᵢ) φ(xᵢ)ᵀ from finite training rows.\n"
             "Resets n_streamed.")
        .def("update",
             [](PyDMKDE& self, NPArrayD x, double alpha){
                 check_row(x, self.m.input_dim(), "x");
                 const double* p = x.data();
                 const size_t n = (size_t)x.shape(0);
                 write_locked(self, [&]{ self.m.update(p, n, alpha); return 0; });
             },
             py::arg("x"), py::arg("alpha"),
             "Streaming rank-1 EMA update: ρ ← (1-α)ρ + α φ(x) φ(x)ᵀ.\n"
             "Raises before fit(), for non-finite x, or α outside (0, 1].")
        .def("score",
             [](const PyDMKDE& self, NPArrayD x){
                 check_row(x, self.m.input_dim(), "x");
                 const double* p = x.data();
                 const size_t n = (size_t)x.shape(0);
                 return read_locked(self, [&]{ return self.m.score(p, n); });
             },
             py::arg("x"),
             "Born-rule density score for one finite row; higher = more 'normal'.")
        .def("score_batch",
             [](const PyDMKDE& self, NPArrayD X){
                 check_width(X, self.m.input_dim(), "X");
                 const size_t n = (size_t)X.shape(0), d = (size_t)X.shape(1);
                 NPArrayD out((py::ssize_t)n);
                 const double* p = X.data();
                 double* o = out.mutable_data();
                 read_locked(self, [&]{ self.m.score_batch(p, n, d, o); return 0; });
                 return out;
             },
             py::arg("X"),
             "Vectorised score over finite rows; returns a 1-D numpy array.")
        .def_property_readonly("feature_dim", [](const PyDMKDE& s){ return s.m.feature_dim(); })
        .def_property_readonly("input_dim",   [](const PyDMKDE& s){ return s.m.input_dim(); })
        .def_property_readonly("sigma",       [](const PyDMKDE& s){ return s.m.sigma(); })
        .def_property_readonly("seed",        [](const PyDMKDE& s){ return s.m.seed(); })
        .def_property_readonly("allow_large", [](const PyDMKDE& s){ return s.m.allow_large(); })
        .def_property_readonly("fitted",
             [](const PyDMKDE& s){ return read_locked(s, [&]{ return s.m.fitted(); }); })
        .def_property_readonly("trace",
             [](const PyDMKDE& s){ return read_locked(s, [&]{ return s.m.trace(); }); })
        .def_property_readonly("n_fitted",
             [](const PyDMKDE& s){ return read_locked(s, [&]{ return s.m.n_fitted(); }); })
        .def_property_readonly("n_streamed",
             [](const PyDMKDE& s){ return read_locked(s, [&]{ return s.m.n_streamed(); }); })
        .def_property_readonly("revision",
             [](const PyDMKDE& s){ return read_locked(s, [&]{ return s.m.revision(); }); },
             "Counter incremented on every change of ρ (fit, update, state import).")
        .def("rho_matrix",
             [](const PyDMKDE& self){
                 const py::ssize_t D = self.m.feature_dim();
                 NPArrayD out({D, D});
                 double* o = out.mutable_data();
                 read_locked(self, [&]{
                     const auto& flat = self.m.rho();
                     std::copy(flat.begin(), flat.end(), o);
                     return 0;
                 });
                 return out;
             },
             "Return a copy of ρ as a (D, D) numpy array.")
        .def("weights",
             [](const PyDMKDE& self){
                 return matrix_copy(self.m.weights(), self.m.feature_dim() / 2, self.m.input_dim());
             },
             "Return a copy of the RFF frequencies W as a (D/2, input_dim) array.")
        .def("transform",
             [](const PyDMKDE& self, NPArrayD x){
                 check_row(x, self.m.input_dim(), "x");
                 const py::ssize_t D = self.m.feature_dim();
                 NPArrayD out(D);
                 self.m.transform(x.data(), (size_t)x.shape(0), out.mutable_data(), (size_t)D);
                 return out;
             },
             py::arg("x"),
             "Return the RFF embedding φ(x) ∈ R^D for a single row.")
        .def("get_state",
             [](const PyDMKDE& self){
                 dmkde::DMKDEState s = read_locked(self, [&]{ return self.m.state(); });
                 return state_to_dict(s, self.m.allow_large());
             },
             "Export the full model state (dims, sigma, seed, W, ρ, counters) as a dict.\n"
             "W is included so a restored model is identical on any platform.")
        .def_static("from_state",
             [](const py::dict& d){
                 bool allow_large = false;
                 dmkde::DMKDEState s = dict_to_state(d, allow_large);
                 return new PyDMKDE(dmkde::DMKDE::from_state(s, allow_large));
             },
             py::arg("state"),
             "Rebuild a model from get_state() output. Validates every field.");

    py::class_<PyMahalanobis>(m, "Mahalanobis",
        "Mahalanobis distance baseline (negative distance = higher = more 'normal').\n"
        "Needs >= 2 finite training rows. Ridge = 1e-6 * trace(C)/d (scale-invariant).")
        .def(py::init<>())
        .def("fit",
             [](PyMahalanobis& self, NPArrayD X){
                 check_2d(X, "X");
                 const double* p = X.data();
                 const size_t n = (size_t)X.shape(0), d = (size_t)X.shape(1);
                 write_locked(self, [&]{ self.m.fit(p, n, d); return 0; });
             },
             py::arg("X"))
        .def_property_readonly("input_dim",
             [](const PyMahalanobis& s){ return read_locked(s, [&]{ return s.m.input_dim(); }); })
        .def_property_readonly("fitted",
             [](const PyMahalanobis& s){ return read_locked(s, [&]{ return s.m.fitted(); }); })
        .def("get_state",
             [](const PyMahalanobis& self){
                 vector<double> mean, P; int d = 0;
                 read_locked(self, [&]{
                     if (!self.m.fitted()) throw std::logic_error("Mahalanobis is not fitted");
                     mean = self.m.mean(); P = self.m.inv_cov(); d = self.m.input_dim();
                     return 0;
                 });
                 py::dict st;
                 st["format"]  = 1;
                 NPArrayD mean_arr((py::ssize_t)d);
                 std::copy(mean.begin(), mean.end(), mean_arr.mutable_data());
                 st["mean"]    = mean_arr;
                 st["inv_cov"] = matrix_copy(P, d, d);
                 return st;
             },
             "Export mean and inverse covariance as a dict.")
        .def_static("from_state",
             [](const py::dict& st){
                 if (!st.contains("format") || st["format"].cast<int>() != 1)
                     throw std::invalid_argument("unsupported Mahalanobis state format");
                 NPArrayD mean = NPArrayD::ensure(st["mean"]);
                 NPArrayD P    = NPArrayD::ensure(st["inv_cov"]);
                 if (!mean || !P || mean.ndim() != 1 || P.ndim() != 2 ||
                     P.shape(0) != mean.shape(0) || P.shape(1) != mean.shape(0))
                     throw std::invalid_argument("mean must be (d,) and inv_cov (d, d)");
                 auto* out = new PyMahalanobis();
                 try {
                     out->m.set_state(vector<double>(mean.data(), mean.data() + mean.size()),
                                      vector<double>(P.data(), P.data() + P.size()));
                 } catch (...) { delete out; throw; }
                 return out;
             },
             py::arg("state"))
        .def("score",
             [](const PyMahalanobis& self, NPArrayD x){
                 if (x.ndim() != 1) throw std::invalid_argument("x must be 1-D");
                 const double* p = x.data();
                 const size_t n = (size_t)x.shape(0);
                 return read_locked(self, [&]{ return self.m.score(p, n); });
             },
             py::arg("x"))
        .def("score_batch",
             [](const PyMahalanobis& self, NPArrayD X){
                 check_2d(X, "X");
                 const size_t n = (size_t)X.shape(0), d = (size_t)X.shape(1);
                 NPArrayD out((py::ssize_t)n);
                 const double* p = X.data();
                 double* o = out.mutable_data();
                 read_locked(self, [&]{ self.m.score_batch(p, n, d, o); return 0; });
                 return out;
             },
             py::arg("X"));

    m.def("roc_auc",
          [](NPArrayD normal, NPArrayD anomaly){
              if (normal.ndim() != 1 || anomaly.ndim() != 1)
                  throw std::invalid_argument("roc_auc expects two 1-D score arrays");
              vector<double> n(normal.data(), normal.data() + normal.size());
              vector<double> a(anomaly.data(), anomaly.data() + anomaly.size());
              py::gil_scoped_release nogil;
              return dmkde::roc_auc(n, a);
          },
          py::arg("normal_scores"), py::arg("anomaly_scores"),
          "Mann-Whitney ROC-AUC; higher 'normal' score is treated as positive.\n"
          "Raises ValueError on empty input or NaN / inf scores.");
}
