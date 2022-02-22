import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from core.variational_simulation import string2U, controlled_gates

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
ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
param = np.array([1.0, 1.0])
# calculate the operation R_k
k, q, i, j = 0, 0, 2, 0
np.array(Operator(string2U(ops[q][j], n_qubits))) - np.array(Operator(controlled_gates(ops[q][j], q, j, n_qubits)))
