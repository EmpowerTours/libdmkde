"""dmkde.qiskit_backend — optional Qiskit cross-check for Born-rule scoring.

For a fitted DMKDE, the test score s(x) = ⟨φ(x)| ρ |φ(x)⟩ is the expectation
value of the Hermitian operator ρ in the state |φ(x)⟩. This module evaluates
the same quantity with ``qiskit.quantum_info``:

  1. φ(x) is computed by the C++ core and renormalised to a unit vector
     (it already has norm 1 up to rounding);
  2. ρ is wrapped as a dense ``qiskit.quantum_info.Operator`` (2ⁿ × 2ⁿ);
  3. the score is ``Statevector(φ(x)).expectation_value(Operator(ρ))``.

This is exact classical statevector simulation. It does not use the
``StatevectorEstimator`` / ``Estimator`` primitives, builds no
``DensityMatrix``, and has no hardware path: no transpilation, shots or error
mitigation. ``to_circuit`` returns a ``StatePreparation`` circuit for |φ(x)⟩
for inspection only; ``score`` does not execute it. It is far slower than the
classical path and exists to cross-check the math.

ρ is re-read from the model whenever the model changed (``fit`` or
``partial_fit``), so scores always reflect the current model.

Usage:
    from dmkde import DMKDE
    from dmkde.qiskit_backend import QiskitDMKDE

    model = DMKDE(feature_dim=16, sigma=1.5).fit(X_train)   # D must be 2^n
    qmodel = QiskitDMKDE(model)
    q_score = qmodel.score(x)
    c_score = model.score_samples(x[None, :])[0]
    assert abs(q_score - c_score) < 1e-10
"""

from __future__ import annotations

import math

import numpy as np

from . import DMKDE, NotFittedError


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
        from qiskit.quantum_info import Operator, Statevector
    except ImportError as e:
        raise ImportError(
            "QiskitDMKDE requires qiskit. Install with `pip install qiskit`."
        ) from e
    return {
        "QuantumCircuit":   QuantumCircuit,
        "StatePreparation": StatePreparation,
        "Operator":         Operator,
        "Statevector":      Statevector,
    }


class QiskitDMKDE:
    """Qiskit statevector cross-check for a fitted DMKDE.

    Parameters
    ----------
    model : DMKDE
        A fitted DMKDE whose ``feature_dim`` is a power of 2.
    """

    def __init__(self, model: DMKDE) -> None:
        if not hasattr(model, "_impl"):
            raise NotFittedError("DMKDE must be fitted before wrapping with QiskitDMKDE")
        D = int(model.feature_dim)
        n = int(math.log2(D)) if D > 0 else 0
        if D <= 0 or (1 << n) != D:
            raise ValueError(
                f"feature_dim={D} must be a power of 2 for amplitude encoding"
            )
        self.model      = model
        self.n_qubits   = n
        self._q         = _require_qiskit()
        self._cache_impl = None   # core object ρ was read from
        self._cache_rev  = None   # its revision counter at that time
        self._rho_op     = None

    def _operator(self):
        """Return Operator(ρ) for the model's current ρ (rebuilt if it changed)."""
        impl = self.model._impl
        rev = impl.revision
        if self._cache_impl is not impl or self._cache_rev != rev:
            rho_arr = np.asarray(impl.rho_matrix(), dtype=np.complex128)
            self._rho_op = self._q["Operator"](rho_arr)
            self._cache_impl = impl
            self._cache_rev = rev
        return self._rho_op

    def encode(self, x):
        """Return the unit-norm amplitude vector φ(x) ∈ R^D as numpy.

        ``x`` must be one finite row with ``model.n_features_in_`` entries.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 1:
            raise ValueError(f"x must be a single row, got shape {x.shape}")
        if x.shape[0] != self.model.n_features_in_:
            raise ValueError(f"x has {x.shape[0]} features, but the model expects "
                             f"{self.model.n_features_in_}")
        if not np.isfinite(x).all():
            raise ValueError("x contains NaN or inf")
        phi = np.asarray(self.model._impl.transform(np.ascontiguousarray(x)),
                         dtype=np.float64)
        n = np.linalg.norm(phi)
        if n < 1e-12:
            raise RuntimeError("RFF embedding produced zero vector")
        return phi / n  # numerical re-normalisation

    def to_circuit(self, x):
        """Return a QuantumCircuit that prepares |φ(x)⟩ on n qubits (not executed)."""
        phi = self.encode(x)
        QC = self._q["QuantumCircuit"]
        SP = self._q["StatePreparation"]
        qc = QC(self.n_qubits)
        qc.append(SP(phi), range(self.n_qubits))
        return qc

    def score(self, x):
        """Compute ⟨φ(x)| ρ |φ(x)⟩ by exact statevector simulation.

        Equal to ``model.score_samples(x[None, :])[0]`` up to rounding, using
        the model's current ρ.
        """
        Statevector = self._q["Statevector"]
        sv = Statevector(self.encode(x))
        ev = sv.expectation_value(self._operator())
        return float(np.real(ev))


__all__ = ["QiskitDMKDE"]
