"""
Lipkin-Meshkov-Glick (LMG) model in full qubit representation.

This module implements the LMG Hamiltonian in the full 2^N dimensional
Hilbert space (N qubits), using the coefficient conventions consistent with
the symmetric-subspace (j=N/2) Hamiltonian implemented in models/Lipkin.py.

The Hamiltonian is:
    Ĥ = -(λ + 2χ²)/4 + Σᵢ [(1/2 - χ²/2N)σ̂ᵢᶻ - (χ/2N)σ̂ᵢˣ]
        - (1/4N)Σᵢ≠ⱼ [λσ̂ᵢˣσ̂ⱼˣ + 2χ(σ̂ᵢˣκ̂ⱼ + κ̂ᵢσ̂ⱼˣ) + 4χ²κ̂ᵢκ̂ⱼ]

where κ̂ᵢ = (σ̂ᵢᶻ + 1)/2

This representation does NOT exploit the permutation symmetry,
so it works in the full 2^N space rather than the (N+1)-dimensional
symmetric subspace.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def pauli_str(N: int, ops: dict) -> str:
    """Create N-qubit Pauli string. Example: pauli(3, {0:'X', 2:'Z'}) -> 'ZIX'"""
    s = ['I'] * N
    for i, p in ops.items():
        s[i] = p
    return ''.join(s)

def build_lipkin_hamiltonian(N: int, lam: float, chi: float) -> SparsePauliOp:
    """Lipkin Hamiltonian matching models/Lipkin.py."""
    terms = []
    
    # Constant 
    const_val = -0.25 * (lam + 2*chi**2)
    terms.append(('I'*N, const_val))
    
    # Single-qubit terms
    c_z = 0.5 * (1 - chi**2/N)
    c_x = -0.5 * chi/N
    
    for i in range(N):
        terms.append((pauli_str(N, {i:'Z'}), c_z))
        terms.append((pauli_str(N, {i:'X'}), c_x))
    
    # Two-qubit terms
    c_xx = -lam / (4*N)
    # The Hamiltonian is specified with a sum over i != j and the symmetric cross-term
    # 2χ( X_i κ_j + κ_i X_j ). The loop below only adds X_i κ_j explicitly (the κ_i X_j
    # contribution is generated when indices are swapped), hence the effective coefficient
    # used here is -χ/N.
    c_xz = -chi / N
    # Likewise, 4χ² κ_i κ_j with prefactor -(1/(4N)) gives an effective coefficient -χ²/N.
    c_zz = -chi**2 / N
    
    for i in range(N):
        for j in range(N):
            if i == j: 
                continue
            # X_i X_j term
            terms.append((pauli_str(N, {i:'X', j:'X'}), c_xx))
            
            # X_i kappa_j where kappa = (Z + I)/2
            # c_xz * X_i * (Z_j + I)/2 = 0.5*c_xz * X_i Z_j + 0.5*c_xz * X_i
            terms.append((pauli_str(N, {i:'X', j:'Z'}), 0.5 * c_xz))
            terms.append((pauli_str(N, {i:'X'}), 0.5 * c_xz))

            # kappa_i kappa_j term
            # c_zz * (Z_i + I)/2 * (Z_j + I)/2 = 0.25*c_zz * (Z_i Z_j + Z_i + Z_j + I)
            terms.append((pauli_str(N, {i:'Z', j:'Z'}), 0.25 * c_zz))
            terms.append((pauli_str(N, {i:'Z'}), 0.5 * c_zz))
            terms.append(('I'*N, 0.25 * c_zz))
            
    return SparsePauliOp.from_list(terms).simplify()

def lipkin_eigenstates(N: int, lam: float, chi: float) -> np.ndarray:
    """
    Compute eigenstates of the Lipkin Hamiltonian for given parameters.
    
    Parameters:
    -----------
    N : int
        Number of qubits
    lam : float
        Interaction strength λ
    chi : float
        Interaction strength χ
    
    Returns:
    --------
    eigvecs : ndarray
        Matrix of eigenvectors (columns)
    """
    H = build_lipkin_hamiltonian(N, lam, chi).to_matrix()
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs

def lipkin_ground_state(N: int, lam: float, chi: float) -> np.ndarray:
    """
    Compute ground state of the Lipkin Hamiltonian for given parameters.
    
    Parameters:
    -----------
    N : int
        Number of qubits
    lam : float
        Interaction strength λ
    chi : float
        Interaction strength χ
    
    Returns:
    --------
    gs : ndarray
        Ground state vector (2^N dimensional)
    """
    return lipkin_eigenstates(N, lam, chi)[:, 0]

def lipkin_spectrum(N: int, lam: float, chi: float) -> np.ndarray:
    """
    Compute energy spectrum of the Lipkin Hamiltonian for given parameters.
    
    Parameters:
    -----------
    N : int
        Number of qubits
    lam : float
        Interaction strength λ
    chi : float
        Interaction strength χ
    
    Returns:
    --------
    spectrum : ndarray
        Array of energy eigenvalues
    """
    H = build_lipkin_hamiltonian(N, lam, chi).to_matrix()
    return np.linalg.eigvalsh(H)