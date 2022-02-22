import pennylane as qml
from pennylane import numpy as np
from core.variational_simulation_pennylane import A_kqij
# create the circuit
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)
# define the parameters of the problem
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([1.0, 1.0])

ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]

k, q, i, j = 0, 1, 2, 1
a_kqij = A_kqij(params, fs, ops, n_qubits, k, q, i, j)
drawer = qml.draw(A_kqij, wire_order=["a", 0, 1, 2])
print(drawer(params, fs, ops, n_qubits, k, q, i, j))
