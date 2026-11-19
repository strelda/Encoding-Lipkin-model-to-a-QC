from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


PAULI_LABELS_1Q: Tuple[str, ...] = ("I", "X", "Y", "Z")


def apply_pauli(qc: QuantumCircuit, qubit: int, p: str) -> None:
    """Apply a 1-qubit Pauli (I/X/Y/Z) onto ``qubit`` in-place."""
    if p == "I":
        return
    if p == "X":
        qc.x(qubit)
        return
    if p == "Y":
        qc.y(qubit)
        return
    if p == "Z":
        qc.z(qubit)
        return
    raise ValueError(f"Unknown 1Q Pauli: {p!r}")


def pauli_commutes(p1: str, p2: str) -> bool:
    """Return True iff two n-qubit Paulis commute.

    Both labels must be same length, using qiskit ordering where the rightmost
    char acts on qubit 0.
    """
    return Pauli(p1).commutes(Pauli(p2))


def pauli_anticommutes(p1: str, p2: str) -> bool:
    return not pauli_commutes(p1, p2)


def random_pauli_1q(rng: np.random.Generator, allowed: Sequence[str] = ("I", "X", "Z")) -> str:
    """Random 1-qubit Pauli from an allowed subset (default: I/X/Z)."""
    allowed = tuple(allowed)
    if not allowed:
        raise ValueError("allowed set must be non-empty")
    return allowed[int(rng.integers(0, len(allowed)))]


def walsh_hadamard_transform(f: np.ndarray) -> np.ndarray:
    """Walsh–Hadamard transform for length 2^m vectors.

    Returns H f / len(f) with H the (unnormalized) Hadamard matrix.

    This is the corrected version of the (non-existent) ``np.fft.hadamard``.
    """
    f = np.asarray(f, dtype=float)
    n = f.shape[0]
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Input length must be a power of 2")

    out = f.copy()
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            a = out[i : i + h].copy()
            b = out[i + h : i + 2 * h].copy()
            out[i : i + h] = a + b
            out[i + h : i + 2 * h] = a - b
        h *= 2

    return out / n


def pauli_tensor_label(paulis: Sequence[str]) -> str:
    """Tensor product label from per-qubit Paulis ordered by qubit index.

    Input is given in qubit-index order (qubit 0 first), but Qiskit Pauli labels
    are big-endian (rightmost char is qubit 0). We therefore reverse.
    """
    for p in paulis:
        if p not in PAULI_LABELS_1Q:
            raise ValueError(f"Invalid Pauli char: {p!r}")
    return "".join(reversed(paulis))


def one_hot_indices(num_qubits: int, dim_h: int) -> List[int]:
    """Indices of one-hot computational basis states |0..010..0> for m=0..dim_h-1."""
    if dim_h < 1:
        raise ValueError("dim_h must be >= 1")
    if dim_h > num_qubits:
        raise ValueError("For one-hot, dim_h must be <= num_qubits")
    return [1 << m for m in range(dim_h)]


def is_one_hot_bitstring(bitstring: str) -> bool:
    """True iff bitstring has Hamming weight exactly 1."""
    return bitstring.count("1") == 1


def post_select_counts(counts: Mapping[str, int], predicate) -> Dict[str, int]:
    """Filter a counts dict by a predicate on bitstring keys."""
    return {b: c for b, c in counts.items() if predicate(b)}
