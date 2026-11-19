"""Partial Pauli twirling for randomized compiling.

For randomized compiling of a CX gate, the canonical trick is:

    (P_in_c ⊗ P_in_t)  →  CX  →  (P_out_c ⊗ P_out_t)

where ``(P_out_c, P_out_t) = CX · (P_in_c, P_in_t) · CX`` is the Heisenberg-
propagation of the input Paulis through CX. With this choice the *net unitary*
equals CX exactly, while the gate noise gets averaged into a stochastic
Pauli channel over many random samples.

If we restrict the input twirl set to the article's partial-twirling alphabet
``{I, X, Z}^⊗2``, the closure-under-CX condition is satisfied by 8 of the 9
possible pairs (the pair ``(X, Z)`` propagates a Y component and is excluded).
That gives a valid partial-twirling set tailored to the [[4,2,2]] error space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit

from .pauli import apply_pauli


# (P_in_c, P_in_t, P_out_c, P_out_t) tuples that close under CX propagation
# when the input Paulis are drawn from {I, X, Z}.
_CX_TWIRL_TABLE: Tuple[Tuple[str, str, str, str], ...] = (
    ("I", "I", "I", "I"),
    ("I", "X", "I", "X"),
    ("I", "Z", "Z", "Z"),
    ("Z", "I", "Z", "I"),
    ("Z", "X", "Z", "X"),
    ("Z", "Z", "I", "Z"),
    ("X", "I", "X", "X"),
    ("X", "X", "X", "I"),
)


@dataclass(frozen=True)
class TwirlingConfig:
    allowed_1q_paulis: Tuple[str, ...] = ("I", "X", "Z")


def _twirl_cx(out: QuantumCircuit, q_c: int, q_t: int, rng: np.random.Generator) -> None:
    """Randomized-compiling twirl of a single CX gate (preserves the net unitary)."""
    pin_c, pin_t, pout_c, pout_t = _CX_TWIRL_TABLE[
        int(rng.integers(0, len(_CX_TWIRL_TABLE)))
    ]
    apply_pauli(out, q_c, pin_c)
    apply_pauli(out, q_t, pin_t)
    out.cx(q_c, q_t)
    apply_pauli(out, q_c, pout_c)
    apply_pauli(out, q_t, pout_t)


def _is_2q_instruction(inst) -> bool:
    try:
        return len(inst.qubits) == 2
    except Exception:
        return False


def apply_partial_twirling(
    qc: QuantumCircuit,
    rng: np.random.Generator,
    config: TwirlingConfig = TwirlingConfig(),
    twirl_2q_only: bool = True,
) -> QuantumCircuit:
    """Return a new circuit where every CX gate has been replaced by a
    properly-propagated randomized-compiling twirl.

    Each CX(c, t) is replaced by ``(P_in_c, P_in_t) · CX · (P_out_c, P_out_t)``
    where ``(P_out_c, P_out_t) = CX · (P_in_c, P_in_t) · CX``. The net unitary
    equals CX exactly; only the gate noise differs.

    Non-CX gates are passed through unchanged.
    """
    out = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    out.metadata = dict(qc.metadata) if qc.metadata else {}

    def _map_qargs(qargs_in):
        return [out.qubits[qc.find_bit(q).index] for q in qargs_in]

    def _map_cargs(cargs_in):
        return [out.clbits[qc.find_bit(c).index] for c in cargs_in]

    for inst, qargs, cargs in qc.data:
        qargs_m = _map_qargs(qargs)
        cargs_m = _map_cargs(cargs)

        if twirl_2q_only and len(qargs_m) == 2 and inst.name.lower() == "cx":
            q_c = qc.find_bit(qargs[0]).index
            q_t = qc.find_bit(qargs[1]).index
            _twirl_cx(out, q_c, q_t, rng)
        else:
            out.append(inst, qargs_m, cargs_m)

    return out
