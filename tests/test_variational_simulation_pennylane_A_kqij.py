import pennylane as qml
from pennylane import numpy as np
from core.variational_simulation_pennylane import A_kqij
import copy
# create the circuit
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)
# define the parameters of the problem
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([1.0, 1.0])


ops = [[qml.PauliZ(0)@qml.PauliZ(1), qml.PauliZ(1)@qml.PauliZ(2),
        qml.PauliZ(0)@qml.PauliZ(2)],
       [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2)]]

# k, q, i, j = 0, 1, 2, 1
# a_kqij = A_kqij(params, fs, ops, n_qubits, k, q, i, j)
# drawer = qml.draw(A_kqij, wire_order=["a", 0, 1, 2])
# print(drawer(params, fs, ops, n_qubits, k, q, i, j))
dev = qml.device("default.qubit", wires=n_qubits)
def ops_ki(ops, n_qubits, k, i):
    ops = ops[:]
    ops[k][i]
    return qml.expval(qml.PauliX(wires="a"))
qml.transforms.get_unitary_matrix(ops_ki)(ops, n_qubits, 0, 1)
# drawer = qml.draw(ops_ki)
# print(drawer(n_qubits, 1, 2))
# ops_ki(ops[0], n_qubits)
# ops_ki(n_qubits, 0)
