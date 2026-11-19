"""Error suppression / mitigation helpers (simulator-only).

Do not import or initialize IBM Runtime / hardware backends here.
"""

from .qedc_422 import (
    STABILIZERS,
    LOGICAL_OPS,
    encode_422,
    decode_422,
    logical_RZ,
    logical_RX,
    measure_decoded,
    post_select_codespace_decoded,
    codespace_survival_decoded,
    logical_expectation_from_counts,
)
from .circuits import exp_pauli
from .pauli import (
    is_one_hot_bitstring,
    post_select_counts,
    pauli_commutes,
    pauli_anticommutes,
    walsh_hadamard_transform,
)
from .pec import (
    PauliChannel,
    is_detectable_by_422,
    reduce_channel_422,
    invert_pauli_channel,
    analytic_pauli_channel_depolarizing,
    pauli_fidelity_through_clifford,
    pauli_fidelity_from_calibration,
    quasiprobability_overhead,
    pec_expectation_from_values,
)
from .propagation import (
    propagate_pauli_through_clifford,
    propagate_channel_through_clifford,
)
from .twirling import (
    TwirlingConfig,
    apply_partial_twirling,
)
from .repetition import (
    BITFLIP_STABILIZERS,
    BITFLIP_LOGICAL_OPS,
    BITFLIP_SYNDROME_TABLE,
    encode_3qubit_bitflip,
    syndrome_3qubit_bitflip,
    decode_3qubit_bitflip,
    apply_correction_3qubit_bitflip,
    majority_vote_logical,
)

__all__ = [
    # qedc_422
    "STABILIZERS",
    "LOGICAL_OPS",
    "encode_422",
    "decode_422",
    "logical_RZ",
    "logical_RX",
    "measure_decoded",
    "post_select_codespace_decoded",
    "codespace_survival_decoded",
    "logical_expectation_from_counts",
    # circuits
    "exp_pauli",
    # pauli
    "is_one_hot_bitstring",
    "post_select_counts",
    "pauli_commutes",
    "pauli_anticommutes",
    "walsh_hadamard_transform",
    # pec
    "PauliChannel",
    "is_detectable_by_422",
    "reduce_channel_422",
    "invert_pauli_channel",
    "analytic_pauli_channel_depolarizing",
    "pauli_fidelity_through_clifford",
    "pauli_fidelity_from_calibration",
    "quasiprobability_overhead",
    "pec_expectation_from_values",
    # propagation
    "propagate_pauli_through_clifford",
    "propagate_channel_through_clifford",
    # twirling
    "TwirlingConfig",
    "apply_partial_twirling",
    # repetition (3-qubit bit-flip)
    "BITFLIP_STABILIZERS",
    "BITFLIP_LOGICAL_OPS",
    "BITFLIP_SYNDROME_TABLE",
    "encode_3qubit_bitflip",
    "syndrome_3qubit_bitflip",
    "decode_3qubit_bitflip",
    "apply_correction_3qubit_bitflip",
    "majority_vote_logical",
]
