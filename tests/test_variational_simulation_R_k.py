import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from core.variational_simulation import R_k
from core.utils import test_R_k

# create the circuit
n_qubits = 3
qr_data = QuantumRegister(n_qubits, "data") # data register
qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
cr = ClassicalRegister(1, "cr") # classical register
qc = QuantumCircuit(qr_data, qr_ancilla, cr)
# define the parameters of the problem
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
operators = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
param = np.array([1.0, 1.0])
# calculate the operation R_k
k = 1
R = R_k(param[0], fs[k], operators[k])
R_test = test_R_k(param[0], fs[k], operators[k], n_qubits)
qc.append(R, qr_data[:])
# print(qc.draw())
R_test - R.to_matrix()
