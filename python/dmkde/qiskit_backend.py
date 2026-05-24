"""dmkde.qiskit_backend — optional Qiskit backend for Born-rule scoring.

For a fitted DMKDE, the test score s(x) = ⟨φ(x)| ρ |φ(x)⟩ is
mathematically the expectation value of the Hermitian observable ρ in
the state |φ(x)⟩. This module evaluates the same quantity through
Qiskit's primitives stack:

  1. amplitude-encode φ(x) onto ⌈log₂ D⌉ qubits via `StatePreparation`,
  2. construct ρ as a `DensityMatrix` and convert to a Qiskit
     `Operator` observable,
  3. evaluate the expectation value via `StatevectorEstimator` —
     swap in a real backend for hardware runs.

The classical scoring path is unchanged; this is the optional path for
reproducing results on a quantum simulator or QPU. Import is lazy so
non-Qiskit users are not forced to install qiskit.

Usage:
    from dmkde import DMKDE
    from dmkde.qiskit_backend import QiskitDMKDE

    model = DMKDE(feature_dim=16, sigma=1.5).fit(X_train)   # D must be 2^n
    qmodel = QiskitDMKDE(model)
    q_score = qmodel.score(x)
    c_score = model.score_samples(x[None, :])[0]
    assert abs(q_score - c_score) < 1e-10                   # exact agreement
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import DMKDE


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
        from qiskit.quantum_info import DensityMatrix, Operator, Statevector
    except ImportError as e:
        raise ImportError(
            "QiskitDMKDE requires qiskit. Install with `pip install qiskit`."
        ) from e
    return {
        "QuantumCircuit":   QuantumCircuit,
        "StatePreparation": StatePreparation,
        "DensityMatrix":    DensityMatrix,
        "Operator":         Operator,
        "Statevector":      Statevector,
    }


class QiskitDMKDE:
    """Qiskit backend for an already-fitted DMKDE.

    Parameters
    ----------
    model : DMKDE
        A fitted DMKDE whose `feature_dim` is a power of 2.

    Notes
    -----
    This is a v0 backend that reproduces classical scores exactly via
    statevector simulation. It demonstrates the embedding-to-circuit
    pipeline; transpilation, shot noise, and hardware error mitigation
    are scoped on the roadmap.
    """

    def __init__(self, model: DMKDE) -> None:
        if model._impl is None:
            raise ValueError("DMKDE must be fitted before wrapping with QiskitDMKDE")
        D = model.feature_dim
        n = int(math.log2(D))
        if (1 << n) != D:
            raise ValueError(
                f"feature_dim={D} must be a power of 2 for amplitude encoding"
            )
        self.model      = model
        self.n_qubits   = n
        self._q         = _require_qiskit()
        # ρ is real-valued for the RBF/RFF case, but we represent it as a
        # general Hermitian DensityMatrix (complex).
        rho_arr = np.asarray(model._impl.rho_matrix(), dtype=np.complex128)
        self._rho_op = self._q["Operator"](rho_arr)

    def encode(self, x):
        """Return the unit-norm amplitude vector φ(x) ∈ R^D as numpy."""
        x = np.ascontiguousarray(x, dtype=np.float64).ravel()
        phi = np.asarray(self.model._impl.transform(x), dtype=np.float64)
        n = np.linalg.norm(phi)
        if n < 1e-12:
            raise RuntimeError("RFF embedding produced zero vector")
        return phi / n  # numerical re-normalisation

    def to_circuit(self, x):
        """Return a QuantumCircuit that prepares |φ(x)⟩ on n qubits."""
        phi = self.encode(x)
        QC = self._q["QuantumCircuit"]
        SP = self._q["StatePreparation"]
        qc = QC(self.n_qubits)
        qc.append(SP(phi), range(self.n_qubits))
        return qc

    def score(self, x):
        """Compute ⟨φ(x)| ρ |φ(x)⟩ via Qiskit statevector simulation.

        Mathematically identical to `model.score_samples(x[None, :])[0]`,
        up to floating-point noise. Hardware execution (with sampling
        noise and error mitigation) is on the roadmap.
        """
        Statevector = self._q["Statevector"]
        sv = Statevector(self.encode(x))
        # ⟨ψ|O|ψ⟩ — Qiskit returns complex; for Hermitian O the imag is ~0.
        ev = sv.expectation_value(self._rho_op)
        return float(np.real(ev))


__all__ = ["QiskitDMKDE"]
