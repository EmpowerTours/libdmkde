"""0.2.0: the RFF weights are the same on every platform, and 0.1.x states still load.

0.1.x drew W with std::normal_distribution, whose output is implementation-defined
(libstdc++, libc++ and MSVC gave different models for one seed). 0.2.0 draws from
dmkde::detail::PortableNormal: std::mt19937_64 (fully specified) and IEEE-754 basic
operations only. These tests pin that sequence three ways:

* golden values, hard-coded below, which must match on every OS;
* an independent pure-Python implementation of the same algorithm (Python floats are
  binary64 and CPython never fuses a multiply-add), compared bit for bit;
* a state and an estimator pickle produced by 0.1.1 (tests/fixtures), which must load
  and score exactly as they did under 0.1.1.
"""
from __future__ import annotations

import json
import platform
import sys
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

import dmkde
from dmkde import _core

FIXTURES = Path(__file__).resolve().parent / "fixtures"
M64 = (1 << 64) - 1


# ---------------------------------------------------------------- reference sampler
class MT19937_64:
    """std::mt19937_64 as specified in [rand.predef] (64-bit Mersenne Twister)."""

    def __init__(self, seed: int) -> None:
        self.mt = [0] * 312
        self.mt[0] = seed & M64
        for i in range(1, 312):
            prev = self.mt[i - 1]
            self.mt[i] = (6364136223846793005 * (prev ^ (prev >> 62)) + i) & M64
        self.i = 312

    def _twist(self) -> None:
        mt = self.mt
        for k in range(312):
            y = (mt[k] & 0xFFFFFFFF80000000) | (mt[(k + 1) % 312] & 0x7FFFFFFF)
            v = mt[(k + 156) % 312] ^ (y >> 1)
            if y & 1:
                v ^= 0xB5026F5AA96619E9
            mt[k] = v
        self.i = 0

    def __call__(self) -> int:
        if self.i >= 312:
            self._twist()
        y = self.mt[self.i]
        self.i += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & M64


def ref_log(x: float) -> float:
    m, e = math.frexp(x)
    if m < 0.70710678118654752440:
        m *= 2.0
        e -= 1
    f = m - 1.0
    t = f / (2.0 + f)
    t2 = t * t
    p = 1.0 / 25.0
    for k in range(11, 0, -1):
        p = 1.0 / float(2 * k + 1) + t2 * p
    two_t = 2.0 * t
    lm = two_t + (two_t * t2) * p
    de = float(e)
    return de * 6.93147180369123816490e-01 + (de * 1.90821492927058770002e-10 + lm)


def ref_normals(seed: int, count: int) -> list[float]:
    g = MT19937_64(seed)
    out: list[float] = []
    while len(out) < count:
        while True:
            u = math.ldexp(float(g() >> 11), -52) - 1.0
            v = math.ldexp(float(g() >> 11), -52) - 1.0
            s = u * u + v * v
            if s < 1.0 and s != 0.0:
                break
        f = math.sqrt(-2.0 * ref_log(s) / s)
        out += [v * f, u * f]
    return out[:count]


def ref_weights(input_dim: int, feature_dim: int, sigma: float, seed: int) -> np.ndarray:
    inv = 1.0 / sigma
    z = ref_normals(seed, (feature_dim // 2) * input_dim)
    return np.array([v * inv for v in z]).reshape(feature_dim // 2, input_dim)


def test_reference_engine_matches_the_standard():
    # [rand.predef]: the 10000th consecutive output of a default-constructed
    # mt19937_64 (seed 5489) is 9981545732273789042.
    g = MT19937_64(5489)
    for _ in range(9999):
        g()
    assert g() == 9981545732273789042


# Golden values: the first six W entries (row-major) built by 0.2.0, as float.hex.
GOLDEN = [
    ((3, 16, 1.5, 42), ["0x1.e145a120e9036p-2", "0x1.b99fc0f1238bep-1", "-0x1.87ea56df1ca5ap-2",
                        "0x1.0faf974dc03e8p-2", "-0x1.45685a9534cafp+0", "0x1.7dccd4f6c61fp-1"]),
    ((5, 8, 1.0, 0), ["0x1.a17559eca5e3fp-4", "-0x1.ece008b6223b9p-2", "-0x1.5c78002cd6723p-1",
                      "0x1.0a30cee39725bp-4", "-0x1.189b3fbdecb34p+0", "0x1.e2e620116a277p+0"]),
    ((2, 64, 0.5, 2**64 - 1), ["0x1.18d13db7ba536p-5", "-0x1.20af0957da38ap+0",
                               "0x1.4e61e83a84a23p-4", "0x1.75fb01cbd5f49p+0",
                               "-0x1.843244fce711ap+0", "-0x1.80f14878712abp+1"]),
]


@pytest.mark.parametrize("args,golden", GOLDEN)
def test_rff_weights_golden_values(args, golden):
    W = _core.DMKDE(*args).weights().ravel()
    assert list(W[:6]) == [float.fromhex(h) for h in golden]


@pytest.mark.parametrize("args", [g[0] for g in GOLDEN] + [
    (1, 2, 3.0, 1), (7, 30, 0.01, 123456789), (4, 256, 2.5, 2**63 + 5), (13, 12, 100.0, 7)])
def test_rff_weights_match_independent_reference_bit_for_bit(args):
    W = _core.DMKDE(*args).weights()
    R = ref_weights(*args)
    assert W.shape == R.shape
    assert W.tobytes() == R.tobytes()


def test_estimator_weights_follow_the_seed():
    X = np.random.default_rng(0).normal(size=(30, 4))
    est = dmkde.DMKDE(feature_dim=32, sigma=2.0, random_state=11).fit(X)
    assert est._impl.weights().tobytes() == ref_weights(4, 32, 2.0, 11).tobytes()


# ---------------------------------------------------------------- 0.1.1 states
def _fixture():
    doc = json.loads((FIXTURES / "dmkde_0_1_1_state.json").read_text(encoding="utf-8"))
    st = dict(doc["state"])
    st["sigma"] = float.fromhex(st["sigma"])
    D, d = st["feature_dim"], st["input_dim"]
    st["W"] = np.array([float.fromhex(v) for v in st["W"]]).reshape(D // 2, d)
    st["rho"] = np.array([float.fromhex(v) for v in st["rho"]]).reshape(D, D)
    Q = np.array([[float.fromhex(v) for v in q] for q in doc["queries"]])
    scores = np.array([float.fromhex(v) for v in doc["scores"]])
    return doc, st, Q, scores



# Saved W and rho load bit-identically everywhere. Scores go through the platform's
# cos (and, on some targets, fused multiply-add), so they are bit-identical only on
# the platform that recorded the fixture (Linux x86_64) and agree to ~1e-12 elsewhere.
_RECORDED_HERE = sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")


def _same_scores(got, want):
    if _RECORDED_HERE:
        assert got.tobytes() == want.tobytes()
    else:
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)

def test_0_1_1_state_loads_and_scores_identically():
    _, st, Q, scores = _fixture()
    m = _core.DMKDE.from_state(st)
    assert m.weights().tobytes() == st["W"].tobytes()
    assert m.rho_matrix().tobytes() == st["rho"].tobytes()
    _same_scores(m.score_batch(Q), scores)
    assert m.n_streamed == 3 and m.n_fitted == 60


def test_0_1_1_state_differs_from_a_fresh_0_2_0_draw():
    # The behaviour change: the same seed now builds different weights.
    _, st, _, _ = _fixture()
    fresh = _core.DMKDE(st["input_dim"], st["feature_dim"], st["sigma"], st["seed"])
    assert fresh.weights().tobytes() != st["W"].tobytes()


def test_0_1_1_estimator_state_dict_loads():
    # What DMKDE.__getstate__ wrote in 0.1.1, rebuilt without pickle (no numpy pin).
    doc, st, Q, scores = _fixture()
    est = dmkde.DMKDE.__new__(dmkde.DMKDE)
    est.__setstate__({"feature_dim": 16, "sigma": 1.5, "random_state": 42,
                      "contamination": 0.05, "allow_large_feature_dim": False,
                      "n_features_in_": 3, "offset_": float.fromhex(doc["offset_"]),
                      "_core_state": st, "_dmkde_state_format": 1})
    _same_scores(est.score_samples(Q), scores)
    assert est.offset_ == float.fromhex(doc["offset_"])


def test_0_1_1_estimator_pickle_loads():
    if int(np.__version__.split(".")[0]) < 2:
        pytest.skip("fixture pickle was written with numpy 2 (numpy._core)")
    doc, _, Q, scores = _fixture()
    est = pickle.loads((FIXTURES / "dmkde_0_1_1_estimator.pkl").read_bytes())
    _same_scores(est.score_samples(Q), scores)
    assert est.offset_ == float.fromhex(doc["offset_"])
    if _RECORDED_HERE:
        assert np.array_equal(est.predict(Q), np.where(scores >= est.offset_, 1, -1))
