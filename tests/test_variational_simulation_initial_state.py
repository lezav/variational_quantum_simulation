import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from core.variational_simulation import initial_state

# create the circuit
n_qubits = 2
qr_data = QuantumRegister(n_qubits, "data") # data register
qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
cr = ClassicalRegister(1, "cr") # classical register
qc = QuantumCircuit(qr_data, qr_ancilla, cr)
# define the parameters of the problem
np.array(Operator(initial_state(n_qubits)))[:, 0]
