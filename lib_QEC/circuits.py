from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
from qiskit import QuantumCircuit


def exp_pauli(qc: QuantumCircuit, pauli_string: str, theta: float, qubits: Sequence[int] | None = None) -> None:
    """Append an n-qubit Pauli exponential exp(-i theta/2 * P).

    This uses a standard construction:
    - basis change to map each non-I Pauli to Z
    - CNOT chain to collect parity onto one target
    - RZ(theta) on target
    - uncompute

    ``pauli_string`` follows Qiskit labeling (rightmost char is qubit 0).
    If ``qubits`` is provided, it maps pauli_string positions onto those circuit qubits.
    """
    if qubits is None:
        if len(pauli_string) != qc.num_qubits:
            raise ValueError("pauli_string length must match qc.num_qubits when qubits is None")
        qubits = list(range(qc.num_qubits))

    if len(pauli_string) != len(qubits):
        raise ValueError("pauli_string length must match len(qubits)")

    # Work in little-endian qubit order for iteration.
    # pauli_string is big-endian: char index -1 is qubit 0.
    chars_by_qubit = list(reversed(pauli_string))

    active: List[int] = [qubits[i] for i, p in enumerate(chars_by_qubit) if p != "I"]
    if not active:
        return

    # basis change
    for i, p in enumerate(chars_by_qubit):
        q = qubits[i]
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)
        elif p in ("Z", "I"):
            pass
        else:
            raise ValueError(f"Invalid Pauli char: {p!r}")

    # parity onto last active qubit
    target = active[-1]
    for control in active[:-1]:
        qc.cx(control, target)

    qc.rz(theta, target)

    # uncompute
    for control in reversed(active[:-1]):
        qc.cx(control, target)

    # undo basis change
    for i, p in enumerate(chars_by_qubit):
        q = qubits[i]
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.h(q)
            qc.s(q)
        elif p in ("Z", "I"):
            pass
