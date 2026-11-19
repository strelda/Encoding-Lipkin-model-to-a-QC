"""[[4,2,2]] QEDC encoder, decoder, logical operators, and measurement helpers.

References
----------
Zhong et al. (2025), Fig. 1 and Sec. IV.

Conventions
-----------
The four data qubits are ordered ``q_data = [q_x, q_z, q_2, q_1]``.
With ``q_data = [0, 1, 2, 3]``, Qiskit produces bitstrings ``b_3 b_2 b_1 b_0``
reading ``q_1 q_2 q_z q_x`` from left to right, so the article's logical basis
matches the natural bitstrings: ``|0̄0̄⟩ = (|0000⟩ + |1111⟩)/√2`` etc.

Logical Pauli operators in Qiskit big-endian labels (leftmost char acts on
the highest-index qubit of ``q_data``):

    X̄_1 = X_{q_z} X_{q_1}  → "XIXI"
    X̄_2 = X_{q_z} X_{q_2}  → "IXXI"
    Z̄_1 = Z_{q_x} Z_{q_1}  → "ZIIZ"
    Z̄_2 = Z_{q_x} Z_{q_2}  → "IZIZ"
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from qiskit import ClassicalRegister, QuantumCircuit

from .circuits import exp_pauli


STABILIZERS: Tuple[str, str] = ("XXXX", "ZZZZ")

LOGICAL_OPS: Dict[str, str] = {
    "X1": "XIXI",
    "X2": "IXXI",
    "Z1": "ZIIZ",
    "Z2": "IZIZ",
}


def encode_422(qc: QuantumCircuit, q_data: Sequence[int]) -> None:
    """[[4,2,2]] encoder (Zhong et al. 2025, Fig. 1 left).

    Maps |0⟩_{q_x} |0⟩_{q_z} |b_2⟩_{q_2} |b_1⟩_{q_1} into the codespace
    state |b̄_1 b̄_2⟩. The H on ``q_x`` is included here, so the caller is
    expected to provide |0⟩ on every data qubit (and the desired logical
    input prepared on ``q_2`` / ``q_1``).
    """
    if len(q_data) != 4:
        raise ValueError("encode_422 expects 4 data qubits")
    q_x, q_z, q_2, q_1 = q_data
    qc.h(q_x)
    qc.cx(q_2, q_z)
    qc.cx(q_1, q_z)
    qc.cx(q_x, q_z)
    qc.cx(q_x, q_2)
    qc.cx(q_x, q_1)


def decode_422(qc: QuantumCircuit, q_data: Sequence[int]) -> None:
    """Inverse of :func:`encode_422` (Zhong et al. 2025, Fig. 1 right).

    Used for QEDC syndrome readout: after decoding, ``q_x`` and ``q_z`` carry
    the syndrome (both must measure 0 to keep the shot under QEDC
    post-selection), and ``(q_2, q_1)`` hold the decoded logical state in
    the Z basis.
    """
    if len(q_data) != 4:
        raise ValueError("decode_422 expects 4 data qubits")
    q_x, q_z, q_2, q_1 = q_data
    qc.cx(q_x, q_1)
    qc.cx(q_x, q_2)
    qc.cx(q_x, q_z)
    qc.cx(q_1, q_z)
    qc.cx(q_2, q_z)
    qc.h(q_x)


def logical_RZ(qc: QuantumCircuit, j: int, theta: float, q_data: Sequence[int]) -> None:
    """Logical Z rotation: exp(-i θ/2 · Z̄_j)."""
    if j not in (1, 2):
        raise ValueError(f"logical qubit j must be 1 or 2, got {j}")
    exp_pauli(qc, LOGICAL_OPS[f"Z{j}"], theta, qubits=list(q_data))


def logical_RX(qc: QuantumCircuit, j: int, theta: float, q_data: Sequence[int]) -> None:
    """Logical X rotation: exp(-i θ/2 · X̄_j)."""
    if j not in (1, 2):
        raise ValueError(f"logical qubit j must be 1 or 2, got {j}")
    exp_pauli(qc, LOGICAL_OPS[f"X{j}"], theta, qubits=list(q_data))


def measure_decoded(
    qc: QuantumCircuit,
    q_data: Sequence[int],
    creg: Optional[ClassicalRegister] = None,
    basis: str = "ZZ",
) -> ClassicalRegister:
    """Append decoder + basis rotation + measurement of all four qubits.

    Classical register layout:
      clbit 0 = q_x (syndrome X)
      clbit 1 = q_z (syndrome Z)
      clbit 2 = q_2 (logical qubit 2)
      clbit 3 = q_1 (logical qubit 1)

    ``basis[0]`` selects the measurement basis for logical qubit 2 and
    ``basis[1]`` for logical qubit 1, both in {"Z","X"}.
    """
    if len(q_data) != 4:
        raise ValueError("measure_decoded expects 4 data qubits")
    if len(basis) != 2 or any(ch not in ("Z", "X") for ch in basis):
        raise ValueError(f"basis must be a 2-char string in {{Z,X}}; got {basis!r}")

    decode_422(qc, q_data)
    q_x, q_z, q_2, q_1 = q_data

    if basis[0] == "X":
        qc.h(q_2)
    if basis[1] == "X":
        qc.h(q_1)

    if creg is None:
        creg = ClassicalRegister(4, "c_decode")
        qc.add_register(creg)
    qc.measure(q_x, creg[0])
    qc.measure(q_z, creg[1])
    qc.measure(q_2, creg[2])
    qc.measure(q_1, creg[3])
    return creg


def _extract_decoded_register(bitstring: str, creg_size: int = 4) -> str:
    """Return the ``creg_size`` chars of the *last-added* register.

    Qiskit places later-added classical registers to the LEFT of the bitstring.
    If the circuit has only one register, this is just the full string. The
    function is robust to either form (spaces, or a single concatenated string).
    """
    if " " in bitstring:
        groups = [g for g in bitstring.split(" ") if g]
        if len(groups) == 1:
            return groups[0]
        # last-added register is leftmost group
        return groups[0]
    return bitstring[:creg_size] if len(bitstring) >= creg_size else bitstring


def post_select_codespace_decoded(
    counts: Mapping[str, int],
    creg_size: int = 4,
) -> Dict[str, int]:
    """Keep only shots whose syndrome (clbits 0 and 1) are both 0.

    Assumes the decoded register layout produced by :func:`measure_decoded`.
    The decoded register must be the LAST register added to the circuit
    (Qiskit places it at the left of the bitstring).
    """
    keep: Dict[str, int] = {}
    for bitstring, count in counts.items():
        reg = _extract_decoded_register(bitstring, creg_size=creg_size)
        if len(reg) < 2:
            continue
        # within reg: leftmost = highest clbit, rightmost = clbit 0
        s_x = reg[-1]  # clbit 0
        s_z = reg[-2]  # clbit 1
        if s_x == "0" and s_z == "0":
            keep[bitstring] = count
    return keep


def codespace_survival_decoded(counts: Mapping[str, int], creg_size: int = 4) -> float:
    """Fraction of shots that survive QEDC post-selection on the decoded readout."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    kept = sum(post_select_codespace_decoded(counts, creg_size=creg_size).values())
    return kept / total


_LOGICAL_OBSERVABLES = {"Z1", "Z2", "Z1Z2", "X1", "X2", "X1X2"}


def logical_expectation_from_counts(
    post_counts: Mapping[str, int],
    observable: str,
    creg_size: int = 4,
) -> float:
    """Expectation of a logical Pauli observable from post-selected counts.

    Supported observables: ``Z1, Z2, Z1Z2, X1, X2, X1X2``.

    For X-basis observables, the measurement circuit must have been built with
    the corresponding logical qubit Hadamard-rotated before measurement (see
    ``basis`` argument of :func:`measure_decoded`).
    """
    if observable not in _LOGICAL_OBSERVABLES:
        raise ValueError(f"Unsupported observable: {observable}")
    total = sum(post_counts.values())
    if total == 0:
        return float("nan")
    acc = 0
    for bitstring, count in post_counts.items():
        reg = _extract_decoded_register(bitstring, creg_size=creg_size)
        # within reg: leftmost = clbit 3, rightmost = clbit 0
        b_q1 = int(reg[0])  # clbit 3 = q_1
        b_q2 = int(reg[1])  # clbit 2 = q_2
        if observable in ("Z1", "X1"):
            eigen = 1 - 2 * b_q1
        elif observable in ("Z2", "X2"):
            eigen = 1 - 2 * b_q2
        else:  # "Z1Z2" or "X1X2"
            eigen = 1 - 2 * ((b_q1 + b_q2) % 2)
        acc += eigen * count
    return acc / total

