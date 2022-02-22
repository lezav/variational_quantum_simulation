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
ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# ops = [["IZZ", "ZZI", "ZIZ"], ["IIX", "IXI", "XII"]]
k, q, i, j = 0, 1, 2, 0

vector = np.array([ 0.35355339,  0.35355339,
        0.35355339, -0.35355339,
        0.35355339, -0.35355339,
       -0.35355339, -0.35355339]).reshape(8, 1) +1j*0
a_kqij = A_kqij(params, fs, ops, n_qubits, k, q, i, j)
a_kqij_test = test_A_kqij(params, fs, ops, n_qubits, k, q, i, j, vector)

a_kqij, a_kqij_test
# np.sum(Operator(string2U2(ops[q][j], n_qubits).control(num_ctrl_qubits=1)) - Operator(string2U(ops[q][j], n_qubits).control(num_ctrl_qubits=1)))

# n_qubits = 2
# J = 1j*1/2
# B = 1j*1/2
# fs = [[-J], [-B, -B]]
# params = np.array([1.0, 1.0])
# ops = [["ZZ"], ["XI", "IX"]]
# # ops = [["IZZ", "ZZI", "ZIZ"], ["IIX", "IXI", "XII"]]
# k, q, i, j = 1, 1, 0, 1
# vector = np.array([0.5, 0.5, 0.5, 0.5]).reshape(4, 1) +1j*0
# a_kqij = A_kqij(params, fs, ops, n_qubits, k, q, i, j)
# a_kqij_test = test_A_kqij(params, fs, ops, n_qubits, k, q, i, j, vector)
#
# a_kqij, a_kqij_test
