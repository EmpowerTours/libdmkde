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
//     phase-fair reader/writer lock (concurrent scoring, exclusive
//     fit/update; read and write phases alternate, so continuous scoring
//     cannot starve update()/partial_fit and busy writers cannot starve
//     scoring);
//   * malformed state dicts raise ValueError, never a pybind11 cast error.

#include "dmkde.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using std::vector;

using NPArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

// Phase-fair reader/writer lock (Brandenburg & Anderson, 2009): read and
// write phases alternate, so neither side can starve the other.
//   * Writers are served in FIFO ticket order. Once a writer holds or waits
//     for a ticket, newly arriving readers queue instead of joining the
//     current read phase, so a writer waits only for the readers already
//     inside (continuous scoring cannot starve update()/partial_fit).
//   * When a writer releases the lock it admits every reader queued so far
//     as one read phase, before the next writer may enter. A reader
//     therefore waits for at most one write critical section (plus the read
//     phase ahead of it), however many writers are busy. The 0.1.1 RC lock
//     let back-to-back writers hand the lock to each other and starved
//     readers for seconds.
// std::shared_mutex in libstdc++ prefers readers (writers can starve).
// Never held together with the GIL (see read_locked / write_locked). Not
// recursive.
class RWLock {
public:
    void lock_shared() {
        std::unique_lock<std::mutex> lk(m_);
        if (w_next_ == w_serve_) {  // no writer active or waiting
            ++readers_;
            return;
        }
        // Wait for the end of the current/next write phase; the releasing
        // writer counts us into readers_ before waking us.
        ++queued_readers_;
        const uint64_t phase = read_phase_;
        cv_.wait(lk, [&]{ return read_phase_ != phase; });
    }
    void unlock_shared() {
        std::lock_guard<std::mutex> lk(m_);
        if (--readers_ == 0) cv_.notify_all();
    }
    void lock() {
        std::unique_lock<std::mutex> lk(m_);
        const uint64_t ticket = w_next_++;
        cv_.wait(lk, [&]{ return w_serve_ == ticket && readers_ == 0; });
    }
    void unlock() {
        std::lock_guard<std::mutex> lk(m_);
        ++w_serve_;
        if (queued_readers_ > 0) {  // start a read phase for everyone queued
            readers_ += queued_readers_;
            queued_readers_ = 0;
            ++read_phase_;
        }
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    long readers_ = 0;           // readers inside (or admitted to) the lock
    long queued_readers_ = 0;    // readers waiting for the next read phase
    uint64_t read_phase_ = 0;    // bumped when a writer admits queued readers
    uint64_t w_next_ = 0;        // next writer ticket to hand out
    uint64_t w_serve_ = 0;       // ticket allowed to hold the write lock
};

// RAII guards for RWLock.
struct ReadGuard {
    RWLock& l;
    explicit ReadGuard(RWLock& x) : l(x) { l.lock_shared(); }
    ~ReadGuard() { l.unlock_shared(); }
    ReadGuard(const ReadGuard&) = delete;
    ReadGuard& operator=(const ReadGuard&) = delete;
};
struct WriteGuard {
    RWLock& l;
    explicit WriteGuard(RWLock& x) : l(x) { l.lock(); }
    ~WriteGuard() { l.unlock(); }
    WriteGuard(const WriteGuard&) = delete;
    WriteGuard& operator=(const WriteGuard&) = delete;
};

// DMKDE plus the lock that makes it safe to call from several Python
// threads once the GIL is released.
struct PyDMKDE {
    dmkde::DMKDE m;
    mutable RWLock mu;

    PyDMKDE(int input_dim, int feature_dim, double sigma, uint64_t seed, bool allow_large)
        : m(input_dim, feature_dim, sigma, seed, allow_large) {}
    explicit PyDMKDE(dmkde::DMKDE&& mm) : m(std::move(mm)) {}
};

struct PyMahalanobis {
    dmkde::Mahalanobis m;
    mutable RWLock mu;
};

// Release the GIL, then take the object's lock (in that order, so a thread
// waiting for the lock never holds the GIL).
template <class Obj, class F>
auto read_locked(const Obj& o, F&& f) {
    py::gil_scoped_release nogil;
    ReadGuard lk(o.mu);
    return f();
}

template <class Obj, class F>
auto write_locked(Obj& o, F&& f) {
    py::gil_scoped_release nogil;
    WriteGuard lk(o.mu);
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

// ---- strict field readers for state dicts (GIL held) -------------------
// Every malformed value raises std::invalid_argument (ValueError), never a
// pybind11 cast error (RuntimeError).

[[noreturn]] void bad_field(const char* key, const char* want, const py::handle& v) {
    std::string tn = py::str(py::type::handle_of(v).attr("__name__"));
    throw std::invalid_argument(std::string("state field '") + key + "' must be " + want +
                                ", got " + tn);
}

py::dict require_dict(const py::object& o, const char* what) {
    if (!py::isinstance<py::dict>(o))
        throw std::invalid_argument(std::string(what) + " state must be a dict");
    return py::reinterpret_borrow<py::dict>(o);
}

py::object get_field(const py::dict& d, const char* key) {
    if (!d.contains(key)) throw std::invalid_argument(std::string("state is missing '") + key + "'");
    return d[key];
}

bool is_bool_like(const py::handle& v) {
    if (PyBool_Check(v.ptr())) return true;
    py::object np_bool = py::module_::import("numpy").attr("bool_");
    return py::isinstance(v, np_bool);
}

// Integer (Python int or numpy integer, not bool, not float) in [lo, hi].
long long get_int(const py::dict& d, const char* key, long long lo, long long hi) {
    py::object v = get_field(d, key);
    if (is_bool_like(v)) bad_field(key, "an integer", v);
    PyObject* idx = PyNumber_Index(v.ptr());
    if (!idx) { PyErr_Clear(); bad_field(key, "an integer", v); }
    py::object iv = py::reinterpret_steal<py::object>(idx);
    int overflow = 0;
    long long x = PyLong_AsLongLongAndOverflow(iv.ptr(), &overflow);
    if (x == -1 && PyErr_Occurred()) { PyErr_Clear(); bad_field(key, "an integer", v); }
    if (overflow || x < lo || x > hi)
        throw std::invalid_argument(std::string("state field '") + key + "' is out of range [" +
                                    std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return x;
}

uint64_t get_u64(const py::dict& d, const char* key) {
    py::object v = get_field(d, key);
    if (is_bool_like(v)) bad_field(key, "an integer", v);
    PyObject* idx = PyNumber_Index(v.ptr());
    if (!idx) { PyErr_Clear(); bad_field(key, "an integer", v); }
    py::object iv = py::reinterpret_steal<py::object>(idx);
    unsigned long long x = PyLong_AsUnsignedLongLong(iv.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw std::invalid_argument(std::string("state field '") + key +
                                    "' must be in [0, 2**64)");
    }
    return (uint64_t)x;
}

double get_real(const py::dict& d, const char* key) {
    py::object v = get_field(d, key);
    if (is_bool_like(v) || !PyNumber_Check(v.ptr()) || PyComplex_Check(v.ptr()))
        bad_field(key, "a real number", v);
    PyObject* f = PyNumber_Float(v.ptr());
    if (!f) { PyErr_Clear(); bad_field(key, "a real number", v); }
    const double x = PyFloat_AsDouble(f);
    Py_DECREF(f);
    return x;
}

bool get_bool(const py::dict& d, const char* key, bool dflt, bool required) {
    if (!required && !d.contains(key)) return dflt;
    py::object v = get_field(d, key);
    if (!is_bool_like(v)) bad_field(key, "a bool", v);
    return PyObject_IsTrue(v.ptr()) == 1;
}

void check_format(const py::dict& d) {
    if (get_int(d, "format", LLONG_MIN, LLONG_MAX) != 1)
        throw std::invalid_argument("unsupported state format");
}

// Real-valued float64 array (no complex, strings or objects).
NPArrayD get_array(const py::dict& d, const char* key) {
    py::object v = get_field(d, key);
    py::array a = py::array::ensure(v);
    if (!a) bad_field(key, "a numeric array", v);
    const char kind = a.dtype().kind();
    if (kind != 'f' && kind != 'i' && kind != 'u')
        bad_field(key, "a real numeric array", v);
    NPArrayD out = NPArrayD::ensure(a);
    if (!out) bad_field(key, "a numeric array", v);
    return out;
}

dmkde::DMKDEState dict_to_state(const py::object& obj, bool& allow_large) {
    py::dict d = require_dict(obj, "DMKDE");
    static const char* keys[] = {"format", "input_dim", "feature_dim", "sigma", "seed",
                                 "W", "rho", "n_fitted", "n_streamed", "fitted"};
    for (const char* k : keys)
        if (!d.contains(k)) throw std::invalid_argument(std::string("state is missing '") + k + "'");
    check_format(d);
    dmkde::DMKDEState s;
    s.input_dim   = (int)get_int(d, "input_dim", 1, INT_MAX);
    s.feature_dim = (int)get_int(d, "feature_dim", 1, INT_MAX);
    s.sigma       = get_real(d, "sigma");
    s.seed        = get_u64(d, "seed");
    s.n_fitted    = get_int(d, "n_fitted", 0, LLONG_MAX);
    s.n_streamed  = get_int(d, "n_streamed", 0, LLONG_MAX);
    s.fitted      = get_bool(d, "fitted", false, true);
    allow_large   = get_bool(d, "allow_large", false, false);
    // Validate dims before trusting them for shape checks.
    dmkde::detail::check_dims(s.input_dim, s.feature_dim, allow_large);
    dmkde::detail::check_sigma(s.sigma);
    NPArrayD W   = get_array(d, "W");
    NPArrayD rho = get_array(d, "rho");
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
        .def("update_batch",
             [](PyDMKDE& self, NPArrayD X, double alpha){
                 check_width(X, self.m.input_dim(), "X");
                 const double* p = X.data();
                 const size_t n = (size_t)X.shape(0), d = (size_t)X.shape(1);
                 write_locked(self, [&]{ self.m.update_batch(p, n, d, alpha); return 0; });
             },
             py::arg("X"), py::arg("alpha"),
             "Atomic streaming update with every row of X in order (same result as\n"
             "update() per row). All rows are validated first (finite input and\n"
             "finite RFF embedding); on any error ρ and the counters are unchanged.")
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
             [](const py::object& d){
                 bool allow_large = false;
                 dmkde::DMKDEState s = dict_to_state(d, allow_large);
                 // ρ validation (PSD check is O(D³)) runs without the GIL.
                 py::gil_scoped_release nogil;
                 return new PyDMKDE(dmkde::DMKDE::from_state(s, allow_large));
             },
             py::arg("state"),
             "Rebuild a model from get_state() output. Validates every field\n"
             "(types, ranges, shapes, finite W and ρ, ρ symmetric, trace 1 and\n"
             "positive semidefinite: smallest eigenvalue > -1e-9 * trace, proved by\n"
             "Cholesky for feature_dim <= 2048, a randomised Lanczos probe above);\n"
             "raises ValueError on any inconsistency.");

    py::class_<PyMahalanobis>(m, "Mahalanobis",
        "Mahalanobis distance baseline (negative distance = higher = more 'normal').\n"
        "Needs >= 2 finite training rows. Ridge = 1e-6 * diag(C) (invariant to\n"
        "per-feature rescaling).")
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
             [](const py::object& obj){
                 py::dict st = require_dict(obj, "Mahalanobis");
                 check_format(st);
                 NPArrayD mean = get_array(st, "mean");
                 NPArrayD P    = get_array(st, "inv_cov");
                 if (mean.ndim() != 1 || P.ndim() != 2 ||
                     P.shape(0) != mean.shape(0) || P.shape(1) != mean.shape(0))
                     throw std::invalid_argument("mean must be (d,) and inv_cov (d, d)");
                 vector<double> mv(mean.data(), mean.data() + mean.size());
                 vector<double> Pv(P.data(), P.data() + P.size());
                 auto* out = new PyMahalanobis();
                 try {
                     // inv_cov validation (Cholesky, O(d³)) runs without the GIL.
                     py::gil_scoped_release nogil;
                     out->m.set_state(std::move(mv), std::move(Pv));
                 } catch (...) { delete out; throw; }
                 return out;
             },
             py::arg("state"),
             "Rebuild from get_state() output. Requires finite mean and a finite,\n"
             "symmetric, positive-definite inv_cov; raises ValueError otherwise.")
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
