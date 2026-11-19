"""Pauli error propagation utilities.

The single-Pauli, single-Clifford-circuit case is just ``Pauli.evolve(Clifford,
frame='s')``. This module also exposes a multi-Pauli helper that propagates
each entry of a Pauli-channel dictionary through a Clifford circuit and
aggregates the results into a new dictionary keyed by the propagated label.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli


def propagate_pauli_through_clifford(pauli_label: str, clifford_circuit: QuantumCircuit) -> str:
    """Compute the label of ``U P U†`` for a Clifford ``U``.

    Raises:
        ValueError: if ``clifford_circuit`` is not Clifford-representable.
    """
    try:
        cliff = Clifford(clifford_circuit)
    except Exception as exc:  # pragma: no cover
        raise ValueError("Circuit is not Clifford-representable") from exc
    return Pauli(pauli_label).evolve(cliff, frame="s").to_label()


def propagate_channel_through_clifford(
    channel: Mapping[str, float],
    clifford_circuit: QuantumCircuit,
) -> Dict[str, float]:
    """Propagate every term of a Pauli channel forward through a Clifford layer.

    For a Pauli channel ``N(ρ) = Σ_j c_j P_j ρ P_j`` and Clifford ``U``,
    the composed channel ``U ∘ N``'s "Pauli weight" picture has the same
    coefficients but on the propagated Paulis ``U P_j U†``. We strip the
    overall phase that ``Pauli.to_label`` may emit ("+", "-", "i") so the keys
    are pure Pauli strings (only the magnitude matters for a Pauli-channel
    coefficient — the phase squares to +1 in ``P ρ P``).
    """
    cliff = Clifford(clifford_circuit)
    out: Dict[str, float] = {}
    for label, coeff in channel.items():
        p_new = Pauli(label).evolve(cliff, frame="s")
        key = _strip_pauli_phase(p_new.to_label())
        out[key] = out.get(key, 0.0) + float(coeff)
    return out


def _strip_pauli_phase(label: str) -> str:
    """Drop the leading phase prefix (``+``, ``-``, ``i``, ``-i``) from a Pauli label."""
    s = label
    while s and s[0] in ("+", "-", "i"):
        s = s[1:]
    return s
