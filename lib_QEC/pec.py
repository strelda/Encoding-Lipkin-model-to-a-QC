"""Pauli channel utilities for the QEDC+PEC pipeline.

The article (Zhong et al. 2025, Sec. II.D) constructs ``N_tot`` by Pauli error
propagation, drops terms anticommuting with X on the first qubit or Z on the
second qubit (= terms anticommuting with the [[4,2,2]] stabilizers ``XXXX`` /
``ZZZZ``) to get ``N_reduced``, and inverts to apply PEC. For a Pauli-channel
observable, the PEC correction reduces to dividing the noisy expectation by
the corresponding Pauli fidelity ``f_A`` (Sec. III.C).

This module provides:

* ``is_detectable_by_422`` — Step 3b detection condition
* ``reduce_channel_422`` — drop detectable terms
* ``invert_pauli_channel`` — invert a diagonal Pauli channel
* ``analytic_pauli_channel_depolarizing`` — per-gate depolarizing channel
* ``pauli_fidelity_through_clifford`` — analytic ``f_A`` for an observable
  Heisenberg-propagated through a Clifford circuit with per-gate depolarizing
  noise
* ``pauli_fidelity_from_calibration`` — empirical ``f_A`` from a calibration sweep
* ``quasiprobability_overhead`` — γ² from the inverse channel
* ``pec_expectation_from_values`` — PEC sum from per-Pauli measurements
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli

from .pauli import pauli_anticommutes, walsh_hadamard_transform


PauliChannel = Dict[str, float]


# ---------------------------------------------------------------------------
# QEDC detection / reduction
# ---------------------------------------------------------------------------


def is_detectable_by_422(pauli_label: str) -> bool:
    """A 4-qubit Pauli is detectable iff it anticommutes with ``XXXX`` or ``ZZZZ``.

    Equivalently (article Step 3b): the term contains an X or Y on the first
    qubit (anticommutes with Z on q_x) or a Z or Y on the second qubit
    (anticommutes with X on q_z) — after un-decoding the Z/X meaning maps to
    the article's qubit labels.
    """
    for stab in ("XXXX", "ZZZZ"):
        if pauli_anticommutes(pauli_label, stab):
            return True
    return False


def reduce_channel_422(channel: Mapping[str, float]) -> PauliChannel:
    """Drop detectable Paulis (idealized QEDC post-selection model)."""
    return {p: c for p, c in channel.items() if not is_detectable_by_422(p)}


def invert_pauli_channel(channel: Mapping[str, float], *, eps: float = 1e-12) -> PauliChannel:
    """Invert a diagonal Pauli channel. ``c_p → 1 / c_p``, clipped at ``eps``."""
    inv: PauliChannel = {}
    for p, c in channel.items():
        inv[p] = 1.0 / c if abs(c) >= eps else 0.0
    return inv


# ---------------------------------------------------------------------------
# Analytic noise channels
# ---------------------------------------------------------------------------


_PAULIS_1Q: Tuple[str, ...] = ("I", "X", "Y", "Z")


def analytic_pauli_channel_depolarizing(p: float, n_qubits: int = 2) -> PauliChannel:
    """Pauli-channel representation of an ``n_qubits``-qubit depolarizing channel.

    Matches Qiskit's ``depolarizing_error(p, n)`` convention: with probability
    ``1 - p`` the channel does nothing, and with probability ``p`` it applies a
    uniformly random non-identity n-qubit Pauli.

    Returns a dict keyed by n-qubit Pauli labels (Qiskit big-endian).
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    n_paulis = 4 ** n_qubits
    weight_nontriv = p / (n_paulis - 1) if n_paulis > 1 else 0.0
    channel: PauliChannel = {}

    def gen(prefix: str, k: int):
        if k == 0:
            channel[prefix] = (1.0 - p) if all(ch == "I" for ch in prefix) else weight_nontriv
            return
        for ch in _PAULIS_1Q:
            gen(prefix + ch, k - 1)

    gen("", n_qubits)
    return channel


def pauli_fidelity_through_clifford(
    observable_label: str,
    circuit: QuantumCircuit,
    p_1q: float = 0.0,
    p_2q: float = 0.0,
) -> float:
    """Analytic Pauli fidelity ``f_A = ⟨A⟩_noisy / ⟨A⟩_ideal`` for ``circuit``.

    Heisenberg-propagates ``A`` backwards through each gate (assumed Clifford or
    safely treated as identity for the propagation), and at each gate multiplies
    by the Pauli fidelity of that gate's depolarizing noise. The per-gate
    multipliers are:

    * 2-qubit depolarizing (rate ``p_2q``, 15 non-identity 2Q Paulis):
      ``1 - 16 p_2q / 15`` if the propagated observable is non-identity on the
      gate's two qubits, else ``1``.
    * 1-qubit depolarizing (rate ``p_1q``, 3 non-identity 1Q Paulis):
      ``1 - 4 p_1q / 3`` if the propagated observable is non-identity on the
      gate's qubit, else ``1``.

    Non-Clifford rotations (``rz``, ``rx``, ``ry``, ``rzz``, …) are treated as
    identity for the propagation step: this is exact when the observable
    commutes with the rotation generator, and is a good approximation
    otherwise (matches the article's "diagonal noise model" used in Sec. III.C
    of Zhong et al. 2025).
    """
    n = circuit.num_qubits
    if len(observable_label) != n:
        raise ValueError(
            f"observable label length {len(observable_label)} must match "
            f"circuit qubit count {n}"
        )

    current = Pauli(observable_label)
    f_total = 1.0
    f_2q = 1.0 - 16.0 * p_2q / 15.0
    f_1q = 1.0 - 4.0 * p_1q / 3.0

    x = current.x.copy()
    z = current.z.copy()

    for circuit_instruction in reversed(circuit.data):
        inst = circuit_instruction.operation
        qargs = circuit_instruction.qubits
        qubit_indices = [circuit.find_bit(q).index for q in qargs]

        if len(qubit_indices) == 2:
            q0, q1 = qubit_indices[0], qubit_indices[1]
            if x[q0] or z[q0] or x[q1] or z[q1]:
                f_total *= f_2q
        elif len(qubit_indices) == 1:
            q0 = qubit_indices[0]
            if x[q0] or z[q0]:
                f_total *= f_1q

        # Heisenberg propagate through the gate (only for Clifford gates).
        gate_name = inst.name.lower()
        if gate_name in _CLIFFORD_GATE_NAMES:
            if gate_name == "cx":
                c, t = qubit_indices[0], qubit_indices[1]
                x[t] ^= x[c]
                z[c] ^= z[t]
            elif gate_name == "h":
                q = qubit_indices[0]
                x[q], z[q] = z[q], x[q]
            elif gate_name in ("s", "sdg"):
                q = qubit_indices[0]
                z[q] ^= x[q]
            elif gate_name in ("sx", "sxdg"):
                q = qubit_indices[0]
                x[q] ^= z[q]
            elif gate_name in ("x", "y", "z", "id"):
                pass
            else:
                step = QuantumCircuit(n)
                step.append(inst, qargs)
                try:
                    cliff = Clifford(step)
                    tmp_pauli = Pauli((z, x)).evolve(cliff, frame="s")
                    x = tmp_pauli.x.copy()
                    z = tmp_pauli.z.copy()
                except Exception:
                    pass

    return f_total


def pauli_fidelity_from_calibration(
    noisy_values: Sequence[float],
    ideal_values: Sequence[float],
    *,
    min_abs_ideal: float = 0.05,
) -> float:
    """Estimate the multiplicative Pauli fidelity from paired observations.

    Solves the 1-parameter least squares ``min_f Σ (n_i - f i_i)²``. Calibration
    points with ``|ideal| < min_abs_ideal`` are skipped so the estimate is not
    dominated by near-zero observables.
    """
    n_vec = np.asarray(noisy_values, dtype=float)
    i_vec = np.asarray(ideal_values, dtype=float)
    if n_vec.shape != i_vec.shape:
        raise ValueError("noisy_values and ideal_values must have the same shape")
    mask = np.abs(i_vec) >= min_abs_ideal
    if not mask.any():
        return float("nan")
    denom = float(np.sum(i_vec[mask] ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(n_vec[mask] * i_vec[mask]) / denom)


# ---------------------------------------------------------------------------
# Older PEC plumbing kept for backward compatibility
# ---------------------------------------------------------------------------


def pauli_transfer_diagonal(channel: Mapping[str, float], pauli_basis: Iterable[str]) -> np.ndarray:
    return np.array([float(channel.get(p, 0.0)) for p in pauli_basis], dtype=float)


def coefficients_from_pauli_fidelities(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("f must be a 1D array")
    return walsh_hadamard_transform(f)


def quasiprobability_overhead(inv_channel: Mapping[str, float]) -> float:
    """γ = Σ_p |w_p| — square it to get the PEC sampling overhead."""
    return float(np.sum(np.abs(list(inv_channel.values()))))


def pec_expectation_from_values(
    values_by_pauli: Mapping[str, float],
    inv_channel: Mapping[str, float],
) -> float:
    """``⟨A⟩_PEC = Σ_p c_p^inv · ⟨P_p A P_p⟩_post``."""
    total = 0.0
    for p, w in inv_channel.items():
        total += float(w) * float(values_by_pauli.get(p, 0.0))
    return total


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_CLIFFORD_GATE_NAMES = frozenset(
    {"x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "id"}
)


def _bare_pauli_label(p: Pauli) -> str:
    """Pauli label with any leading phase prefix (``+``, ``-``, ``i``) stripped."""
    s = p.to_label()
    while s and s[0] in ("+", "-", "i"):
        s = s[1:]
    return s


def _is_identity_on(label: str, qubits: Sequence[int], n: int) -> bool:
    """Return True iff ``label`` is ``I`` on every qubit in ``qubits``.

    ``label`` is a Qiskit big-endian Pauli string (rightmost char = qubit 0).
    """
    for q in qubits:
        if q < 0 or q >= n:
            return False
        if label[n - 1 - q] != "I":
            return False
    return True
