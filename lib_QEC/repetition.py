"""3-qubit bit-flip repetition code.

Encodes one logical qubit into three physical data qubits::

    |0̄⟩ = |000⟩,    |1̄⟩ = |111⟩.

Stabilisers ``Z_0 Z_1`` and ``Z_1 Z_2`` detect any single-qubit bit-flip
(``X``) error and identify which qubit was hit. The code is blind to
``Z`` errors — a single ``Z`` on any data qubit is a logical ``Z̄``.

Public API mirrors :mod:`lib_QEC.qedc_422` so the two codes feel
interchangeable.

Conventions
-----------
Qiskit labels Pauli strings big-endian: the leftmost character acts on
the highest-index qubit. For three data qubits indexed ``[0, 1, 2]``:

* ``Z_0 Z_1`` → ``"IZZ"``  (I on q2, Z on q1, Z on q0)
* ``Z_1 Z_2`` → ``"ZZI"``  (Z on q2, Z on q1, I on q0)

Bitstring measurement results follow the same big-endian rule: the
rightmost character of the result string is clbit 0.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from qiskit import QuantumCircuit


BITFLIP_STABILIZERS: Tuple[str, str] = ("IZZ", "ZZI")

BITFLIP_LOGICAL_OPS: Dict[str, str] = {
    "X": "XXX",   # X̄ = X_0 X_1 X_2 (transversal)
    "Z": "IIZ",   # Z̄ = Z on any one data qubit (here qubit 0)
}

# Syndrome bitstring -> data qubit index that suffered the X error.
# Syndrome key reads big-endian: ``"s1 s0"`` where s_i = ancilla i.
# Ancilla 0 measures Z_0 Z_1, ancilla 1 measures Z_1 Z_2.
BITFLIP_SYNDROME_TABLE: Dict[str, Optional[int]] = {
    "00": None,   # no detected error
    "01": 0,      # only Z_0 Z_1 fired  → X on data qubit 0
    "11": 1,      # both fired          → X on data qubit 1
    "10": 2,      # only Z_1 Z_2 fired  → X on data qubit 2
}


def encode_3qubit_bitflip(qc: QuantumCircuit, data_qubits: Sequence[int]) -> None:
    """Apply the 2-CX encoder to spread the logical state across three qubits.

    The caller must prepare the desired logical state on ``data_qubits[0]``
    and leave the other two in ``|0⟩``. The encoder then implements

        α|0⟩ + β|1⟩  ⊗  |00⟩   →   α|000⟩ + β|111⟩.
    """
    if len(data_qubits) != 3:
        raise ValueError("encode_3qubit_bitflip expects 3 data qubits")
    qc.cx(data_qubits[0], data_qubits[1])
    qc.cx(data_qubits[0], data_qubits[2])


def syndrome_3qubit_bitflip(
    qc: QuantumCircuit,
    data_qubits: Sequence[int],
    ancilla_qubits: Sequence[int],
    clbits: Sequence[int],
) -> None:
    """Extract both stabiliser syndromes onto two ancilla bits.

    ``ancilla_qubits[0]`` reads ``Z_0 Z_1`` (CX from data 0 then data 1).
    ``ancilla_qubits[1]`` reads ``Z_1 Z_2`` (CX from data 1 then data 2).
    Each ancilla is then measured into the corresponding ``clbits[i]``.
    Ancillas are assumed to start in ``|0⟩``.
    """
    if len(data_qubits) != 3:
        raise ValueError("syndrome_3qubit_bitflip expects 3 data qubits")
    if len(ancilla_qubits) != 2 or len(clbits) != 2:
        raise ValueError("syndrome_3qubit_bitflip expects 2 ancillas and 2 clbits")

    qc.cx(data_qubits[0], ancilla_qubits[0])
    qc.cx(data_qubits[1], ancilla_qubits[0])
    qc.cx(data_qubits[1], ancilla_qubits[1])
    qc.cx(data_qubits[2], ancilla_qubits[1])
    qc.measure(ancilla_qubits[0], clbits[0])
    qc.measure(ancilla_qubits[1], clbits[1])


def decode_3qubit_bitflip(qc: QuantumCircuit, data_qubits: Sequence[int]) -> None:
    """Inverse encoder.

    Applies ``CX(0, 2)`` then ``CX(0, 1)`` to undo the encoder. In the
    noise-free limit this leaves the logical state on ``data_qubits[0]``
    and ``|00⟩`` on the other two.
    """
    if len(data_qubits) != 3:
        raise ValueError("decode_3qubit_bitflip expects 3 data qubits")
    qc.cx(data_qubits[0], data_qubits[2])
    qc.cx(data_qubits[0], data_qubits[1])


def apply_correction_3qubit_bitflip(
    counts: Dict[str, int],
    *,
    n_data: int = 3,
    n_syndrome: int = 2,
    syndrome_at_low_clbits: bool = True,
) -> Dict[str, int]:
    """Classically apply the syndrome-based X correction to count keys.

    The circuit is expected to measure ``n_data`` data bits and
    ``n_syndrome`` syndrome bits into the same classical register.
    When ``syndrome_at_low_clbits=True`` the syndrome lives in the
    lowest-index clbits, so each Qiskit count key looks like

        "d_{n-1} … d_1 d_0  s_1 s_0"        (no spaces in the actual key)

    i.e. the rightmost ``n_syndrome`` characters are the syndrome.

    Returns counts keyed by the *data* bitstring only (length ``n_data``,
    big-endian), aggregated over syndrome outcomes after correction.
    """
    corrected: Dict[str, int] = {}
    for key, c in counts.items():
        bits = key.replace(" ", "")
        if syndrome_at_low_clbits:
            data_part = bits[:n_data]
            syndrome_part = bits[n_data : n_data + n_syndrome]
        else:
            data_part = bits[n_syndrome : n_syndrome + n_data]
            syndrome_part = bits[:n_syndrome]
        flip_idx = BITFLIP_SYNDROME_TABLE.get(syndrome_part)
        if flip_idx is not None:
            pos_in_str = n_data - 1 - flip_idx
            ch = data_part[pos_in_str]
            flipped = "1" if ch == "0" else "0"
            data_part = data_part[:pos_in_str] + flipped + data_part[pos_in_str + 1 :]
        corrected[data_part] = corrected.get(data_part, 0) + c
    return corrected


def majority_vote_logical(corrected_counts: Dict[str, int]) -> Dict[str, int]:
    """Reduce 3-bit corrected data strings to a single logical bit by majority.

    Returns a dict ``{"0": n0, "1": n1}`` summing shots from all keys.
    """
    out = {"0": 0, "1": 0}
    for bits, c in corrected_counts.items():
        n_ones = sum(1 for b in bits if b == "1")
        logical = "1" if n_ones >= 2 else "0"
        out[logical] += c
    return out
