# Quantum State Driving of the Lipkin-Meshkov-Glick Model: Encoding Comparison

Comparison of binary vs. one-hot encoding schemes for quantum state driving of the Lipkin-Meshkov-Glick (LMG) model on gate-based quantum computers.

Includes the full summary PDF in [`paper/State_driving_of_the_Lipkin_Meshkov_Glick_model_on_a_quantum_computer.pdf`](paper/State_driving_of_the_Lipkin_Meshkov_Glick_model_on_a_quantum_computer.pdf).

## Overview

Simulating time-dependent quantum evolution on near-term hardware involves a fundamental trade-off between the number of qubits and circuit depth:

- **Binary Encoding**: Embeds the $(N+1)$-dimensional maximal spin subspace ($j = N/2$) into $N_q = \lceil \log_2(N+1) \rceil$ qubits (e.g., 3 qubits for $N=5$). Minimizes qubit overhead, but generates multi-qubit Pauli terms with deep CNOT synthesis chains.
- **One-Hot Encoding**: Uses $N_q = N+1$ qubits (e.g., 6 qubits for $N=5$) with physical states mapped to computational basis states of Hamming weight 1. The Hamiltonian reduces to $O(N)$ local terms, and hopping terms map directly to 2-qubit $XX+YY$ gates (`XXPlusYYGate`), substantially decreasing circuit depth.

## Key Observations

- **Short Driving Times ($T=1$):** Binary encoding achieves better fidelity. At low depth, gate errors are modest, and one-hot is penalized by higher qubit readout error floors, measurement operator complexity, and amplitude damping ($T_1$) leakage into the non-physical state $|000000\rangle$.
- **Long Driving Times ($T=10$):** One-hot encoding outperforms binary. For longer durations, gate synthesis errors from deep binary circuits dominate, while one-hot benefits from its constant-depth parallel hopping layers.
- **Leakage Mitigation:** Post-selection on Hamming weight 1 removes leakage from $T_1$ decay to the vacuum state, restoring one-hot fidelity.

## Repository Contents

- `paper/`: Document PDF and LaTeX source files.
- `notebooks/`:
  - `01_QC_driving_Lipkin_binary.ipynb`: State driving benchmark using binary encoding.
  - `02_QC_driving_Lipkin_one_hot.ipynb`: State driving benchmark using one-hot encoding.
  - `03_QC_driving_Lipkin_leakage_mitigation.ipynb`: Subspace leakage analysis and post-selection mitigation.
  - `04_QC_driving_spectrum_SQD.ipynb`: Subspace Quantum Diagonalization (SQD) for driving-path spectroscopy.
- `models/`: Implementations of LMG Hamiltonians, spin operators, and one-hot Trotterization circuits.
- `lib_QEC/`: Utilities for Pauli twirling, syndrome extraction, and error mitigation.
- `QC_driving_spectrum_SQD_original_results/`: Cached execution data from hardware experiments.

## Setup & Reproduction

```bash
# Clone the repository
git clone https://github.com/strelda/Encoding-Lipkin-model-to-a-QC.git
cd Encoding-Lipkin-model-to-a-QC

# Create environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebooks
jupyter lab
```

Simulations can run locally using `AerSimulator` or `FakeAachen` without active IBM Quantum credentials.
