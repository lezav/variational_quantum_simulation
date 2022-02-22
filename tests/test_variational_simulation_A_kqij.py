import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from core.variational_simulation import A_kqij
from core.utils import test_A_kqij


n_qubits = 3
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([1.0, 1.0])
ops = [["ZZZ", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
k, q, i, j = 0, 1, 2, 1
vector = np.array([ 0.35355339,  0.35355339, 0.35355339, -0.35355339,
                    0.35355339, -0.35355339, -0.35355339,
                    -0.35355339]).reshape(8, 1) +1j*0
a_kqij = A_kqij(params, fs, ops, n_qubits, k, q, i, j)
a_kqij_test = test_A_kqij(params, fs, ops, n_qubits, k, q, i, j, vector)

a_kqij, a_kqij_test


# n_qubits = 2
# J = 1j*1/2
# B = 1j*1/2
# fs = [[-J, -J, -J], [-B, -B, -B]]
# params = np.array([1.0, 1.0])
# ops = [["ZZ", "IZ", "ZI"], ["XI", "IX", "XX"]]
# k, q, i, j = 0, 1, 1, 0
# # a_kqij = A_kqij(params, fs, ops, n_qubits, k, q, i, j)
# R_ki, R_qj = test_A_kqij(params, fs, ops, n_qubits, k, q, i, j)
# ((R_ki.conj().T) @ R_qj)
