import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from core.variational_simulation import A_kqij, controlled_gates
from core.utils import test_A_kqij


n_qubits = 3
qr_data = QuantumRegister(n_qubits, "data") # data register
qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
cr = ClassicalRegister(1, "cr") # classical register
qc = QuantumCircuit(qr_ancilla, qr_data, cr)

J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([1.0, 1.0])
ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# ops = [["IZZ", "ZZI", "ZIZ"], ["IIX", "IXI", "XII"]]
k, q, i, j = 1, 1, 1, 2
controlled_Uq = controlled_gates(k, i, n_qubits).control(num_ctrl_qubits=1)
qc.append(controlled_Uq, qr_ancilla[:] + qr_data[:])
qc.draw()
