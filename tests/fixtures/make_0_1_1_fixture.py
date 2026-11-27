"""Regenerate the 0.1.1 state fixtures (run ONLY against a dmkde 0.1.1 build).

Produced from commit a8118f3 (dmkde 0.1.1, std::normal_distribution RFF draw on
libstdc++). tests/test_rng_portable.py checks that 0.2.0 still loads these
states and scores them bit-identically, although 0.2.0 draws different
weights for the same seed.

    PYTHONPATH=python python tests/fixtures/make_0_1_1_fixture.py
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np

import dmkde

assert dmkde.__version__ == "0.1.1", dmkde.__version__
HERE = Path(__file__).resolve().parent

rng = np.random.default_rng(20261126)
X = rng.normal(size=(60, 3))
Q = np.vstack([rng.normal(size=(5, 3)), [[4.0, -4.0, 4.0]]])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    est = dmkde.DMKDE(feature_dim=16, sigma=1.5, random_state=42).fit(X)
    est.partial_fit(X[:3], alpha=0.05)

core = est._impl.get_state()


def hexlist(a: np.ndarray) -> list[str]:
    return [float(v).hex() for v in np.asarray(a, dtype=np.float64).ravel()]


doc = {
    "produced_by": "dmkde " + dmkde.__version__ + " (commit a8118f3, libstdc++)",
    "state": {
        "format": int(core["format"]),
        "input_dim": int(core["input_dim"]),
        "feature_dim": int(core["feature_dim"]),
        "sigma": float(core["sigma"]).hex(),
        "seed": int(core["seed"]),
        "allow_large": bool(core["allow_large"]),
        "W": hexlist(core["W"]),
        "rho": hexlist(core["rho"]),
        "n_fitted": int(core["n_fitted"]),
        "n_streamed": int(core["n_streamed"]),
        "fitted": bool(core["fitted"]),
    },
    "offset_": float(est.offset_).hex(),
    "queries": [hexlist(q) for q in Q],
    "scores": hexlist(est.score_samples(Q)),
}
(HERE / "dmkde_0_1_1_state.json").write_text(json.dumps(doc, indent=1) + "\n")
# A real 0.1.1 estimator pickle (protocol 4; needs numpy >= 2 to unpickle).
(HERE / "dmkde_0_1_1_estimator.pkl").write_bytes(pickle.dumps(est, protocol=4))
print("wrote fixtures; scores:", est.score_samples(Q))
