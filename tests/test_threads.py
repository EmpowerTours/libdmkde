"""Thread-safety checks for the GIL-releasing bindings (0.1.1).

The heavy C++ loops run without the GIL under a per-object reader/writer
lock. These tests hammer one model from several threads and check that
results are exactly what a serial run produces.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import dmkde


def test_concurrent_scoring_matches_serial():
    rng = np.random.default_rng(0)
    m = dmkde.DMKDE(feature_dim=256, sigma=1.0).fit(rng.normal(size=(300, 4)))
    chunks = [rng.normal(size=(200, 4)) for _ in range(8)]
    ref = [m.score_samples(c) for c in chunks]
    with ThreadPoolExecutor(4) as ex:
        par = list(ex.map(m.score_samples, chunks))
    for a, b in zip(ref, par):
        np.testing.assert_array_equal(a, b)


def test_one_writer_many_readers_is_consistent():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    stream = rng.normal(size=(200, 4)) + 1.0
    Q = rng.normal(size=(100, 4))

    serial = dmkde.DMKDE(feature_dim=128, sigma=1.0).fit(X)
    for row in stream:
        serial.partial_fit(row[None, :], alpha=0.01)

    m = dmkde.DMKDE(feature_dim=128, sigma=1.0).fit(X)

    def writer():
        for row in stream:
            m.partial_fit(row[None, :], alpha=0.01)

    def reader():
        outs = []
        for _ in range(20):
            s = m.score_samples(Q)
            outs.append(s)
            assert np.all(np.isfinite(s)) and np.all(s >= -1e-12) and np.all(s <= 1 + 1e-9)
            t = m.trace_
            assert abs(t - 1.0) < 1e-9
        return outs

    with ThreadPoolExecutor(4) as ex:
        fw = ex.submit(writer)
        readers = [ex.submit(reader) for _ in range(3)]
        fw.result()
        for f in readers:
            f.result()

    np.testing.assert_array_equal(m._impl.rho_matrix(), serial._impl.rho_matrix())
    assert m._impl.n_streamed == len(stream)


def test_concurrent_fit_and_mahalanobis():
    rng = np.random.default_rng(2)
    data = [rng.normal(size=(500, 6)) for _ in range(4)]

    def job(X):
        a = dmkde.DMKDE(feature_dim=128).fit(X).score_samples(X[:50])
        b = dmkde.Mahalanobis().fit(X).score_samples(X[:50])
        return a, b

    ref = [job(X) for X in data]
    with ThreadPoolExecutor(4) as ex:
        par = list(ex.map(job, data))
    for (a1, b1), (a2, b2) in zip(ref, par):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)
