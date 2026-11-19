# this was rewritten based on Pavel's code in https://github.com/PavelStransky/Geodesic/blob/master/models/LipkinMatus.jl
# only the spectrum was shifted to match our definition in the article
# so that my Mathematica implementation corresponds to it

from typing import Union
import numpy as np
from numpy.typing import NDArray

from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Operator

def shift_plus(N, i):
    """Raising operator J+
    
    Matrix element: sqrt((N - i + 1) * i)
    """
    if i < 0 or i > N:
        return 0.0
    return np.sqrt((N - i + 1) * i)

def shift_minus(N, i):
    """Lowering operator J-
    
    Matrix element: sqrt((i - 1) * (N - i + 2))
    """
    if i <= 0 or i > N + 1:
        return 0.0
    return np.sqrt((i - 1) * (N - i + 2))

def lipkin(N, a=0, jx=0, jz=0, jxx=0, jzz=0, jxz=0):
    """General Lipkin model Hamiltonian
    
    H = a + jx*Jx + jz*Jz + jxx*Jx² + jzz*Jz² + jxz*{Jx,Jz}
    
    where:
    - Jx = 0.5 * (J+ + J-)
    - Jxx = 0.25 * (J+² + J+J- + J-J+ + J-²)
    - Jxz = 0.5 * ({J+,Jz} + {J-,Jz})
    
    Parameters:
    -----------
    N : int
        Total angular momentum quantum number (2j)
    a, jx, jz, jxx, jzz, jxz : float
        Coupling constants
    
    Returns:
    --------
    H : ndarray
        Symmetric Hamiltonian matrix of dimension (N+1) × (N+1)
    """
    dim = N + 1
    result = np.eye(dim, dtype=np.float64)
    
    for i in range(1, dim + 1):  # Julia 1-indexing → Python adjustment
        m = (i - 1) - 0.5 * N  # Magnetic quantum number
        
        # Diagonal terms: a + jz*Jz + jzz*Jz² + jxx*(J+J- + J-J+)/4 - 1.25 (energy shift)
        result[i-1, i-1] = (a - jz * m + jzz * m * m + 
                           0.25 * jxx * (shift_minus(N, i + 1) * shift_plus(N, i) + 
                                        shift_plus(N, i - 1) * shift_minus(N, i)))
        
        # Off-diagonal: nearest neighbor (Jx and Jxz contributions)
        if i > 1:
            r = 0.5 * shift_minus(N, i) * (jx - jxz * (2 * m - 1))
            result[i - 2, i - 1] = r
            result[i - 1, i - 2] = r
        
        if i < dim:
            r = 0.5 * shift_plus(N, i) * (jx - jxz * (2 * m + 1))
            result[i, i - 1] = r
            result[i - 1, i] = r
        
        # Off-diagonal: next-nearest neighbor (Jxx contributions from J+² and J-²)
        if i > 2:
            r = 0.25 * jxx * shift_minus(N, i - 1) * shift_minus(N, i)
            result[i - 3, i - 1] = r
            result[i - 1, i - 3] = r
        
        if i < dim - 1:
            r = 0.25 * jxx * shift_plus(N, i + 1) * shift_plus(N, i)
            result[i + 1, i - 1] = r
            result[i - 1, i + 1] = r
    
    return result

def LipkinHamiltonian(N, λ, χ):
    """Lipkin model for geodesic studies
    
    Specific parameterization with interaction strength λ and deformation χ
    """    
    jx = -χ
    jz = 1 - χ * χ
    jxx = -λ / N
    jxz = -χ / N
    jzz = -χ * χ / N
    a = -0.25 * N * χ * χ  # 0.5 * N * (1 - χ * χ) # Energy shift Pavel had in his code, but we have different shift in the article, so this corresponds to my Mathematica code
    
    return lipkin(N, a=a, jx=jx, jz=jz, jxx=jxx, jxz=jxz, jzz=jzz)

def model_derivative(N, λ, χ, parameters=None):
    """Derivatives ∂H/∂λ and ∂H/∂χ for the Lipkin model
    
    Returns:
    --------
    h1 : ndarray
        ∂H/∂λ
    h2 : ndarray
        ∂H/∂χ
    """
    
    v1 = lipkin(N, jxx=-1/N)
    v2 = lipkin(N, jxz=-1/N, jx=-1)
    v3 = lipkin(N, a=-N/2, jz=-1, jzz=-1/N)
    
    h1 = v1
    h2 = v2 + 2 * χ * v3
    
    return h1, h2

def num_in_nn(nn):
    """Helper function for diabolic points calculation"""
    if nn % 2 == 0:
        return nn // 2
    else:
        return (nn - 1) // 2

def chi_g(nn):
    """χ coordinates for diabolic points"""
    n = num_in_nn(nn)
    if nn % 2 == 0:
        return np.array([np.sqrt(nn / (nn - 1 + 2 * k)) for k in range(1, n + 1)])
    else:
        return np.array([np.sqrt(nn / (nn + 2 * k)) for k in range(1, n + 1)])

def lambda_g(nn):
    """λ coordinates for diabolic points"""
    n = num_in_nn(nn)
    if nn % 2 == 0:
        return np.array([(-nn + 2 * k - 1) / (2 * k - 1) for k in range(1, n + 1)])
    else:
        return np.array([(-nn + 2 * k) / (2 * k) for k in range(1, n + 1)])

def diabolic_points(nn):
    """
    Calculate diabolic points for Lipkin model with N particles
    
    Parameters:
    -----------
    nn : int
        Number of particles (2j)
    
    Returns:
    --------
    points : ndarray
        Array of (λ, χ) coordinates for diabolic points
    """
    lam = lambda_g(nn)
    chi = chi_g(nn)
    return np.column_stack((lam, chi))

def qpt_line(lambda_val: Union[float, NDArray[np.floating]]) -> Union[float, NDArray[np.floating]]:
    """
    QPT (Quantum Phase Transition) line in (λ, χ) space
    
    Parameters:
    -----------
    lambda_val : float or ndarray
        Lambda value(s)
    
    Returns:
    --------
    chi : float or ndarray
        Corresponding χ value(s) on QPT line
    """
    if isinstance(lambda_val, (int, float, np.number)):
        lam = float(lambda_val)
        if lam < 1:
            return float(np.sqrt((lam - 1) / (lam - 2)))
        else:
            return 0.0
    else:
        lambda_arr = np.asarray(lambda_val, dtype=float)
        result = np.zeros_like(lambda_arr)
        mask = lambda_arr < 1
        result[mask] = np.sqrt((lambda_arr[mask] - 1) / (lambda_arr[mask] - 2))
        return result


# export to use as module
__all__ = ['LipkinHamiltonian', 'diabolic_points', 'qpt_line']

# test run
if __name__ == "__main__":
    N = 4
    lam = 1.0
    chi = 0.5
    correct_H = np.array([[0.75, -0.875, -0.306186, 0, 0],
                 [-0.875, -0.1875, -0.765466, -0.375, 0],
                 [-0.306186, -0.765466, -1., -0.459279, -0.306186],
                 [0, -0.375, -0.459279, -1.6875, -0.125],
                 [0, 0, -0.306186, -0.125, -2.25]])
    H_lipkin = LipkinHamiltonian(N, lam, chi)
    # print("Lipkin Hamiltonian for N =", N)
    # print(H_lipkin.real)
    
    dif1 = H_lipkin.real - correct_H

    N = 3
    lam = -2.5
    chi = -1.5
    correct_H = np.array([[-4.625, 2.16506, 0.721688, 0],
                        [2.16506, -1.04167, 1.5, 0.721688],
                        [0.721688, 1.5, 0.208333, 0.433013],
                        [0, 0.721688, 0.433013, -0.875]])
    H_lipkin = LipkinHamiltonian(N, lam, chi)
    # print("Lipkin Hamiltonian for N =", N)
    # print(H_lipkin.real)
    dif2 = H_lipkin.real - correct_H

    N = 6
    lam = 2.5
    chi = 4.3
    correct_H = np.array([[-108.565, -9.65507, -0.806872, 0, 0, 0, 0],
                          [-9.65507, -76.7083, -10.1983, -1.14109, 0, 0, 0],
                          [-0.806872, -10.1983, -50.5983, -8.68912, -1.25, 0, 0],
                          [0, -1.14109, -8.68912, -30.235, -6.20652, -1.14109, 0],
                          [0, 0, -1.25, -6.20652, -15.6183, -3.39945, -0.806872],
                          [0, 0, 0, -1.14109, -3.39945, -6.74833, -0.877734],
                          [0, 0, 0, 0, -0.806872, -0.877734, -3.625]])
    H_lipkin = LipkinHamiltonian(N, lam, chi)
    dif3 = H_lipkin.real - correct_H
    # tests
    tol = 1e-3
    assert np.all(np.abs(dif1) < tol), "Lipkin Hamiltonian test failed for N=4"
    assert np.all(np.abs(dif2) < tol), "Lipkin Hamiltonian test failed for N=3"
    assert np.all(np.abs(dif3) < tol), "Lipkin Hamiltonian test failed for N=6"
    print("All tests passed.")

