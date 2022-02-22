import pennylane as qml
from pennylane import numpy as np
from core.variational_simulation_pennylane import string2op
# create the circuit
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)
# define the parameters of the problem
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([1.0, 1.0])

ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]


ops_ki(ops[1][1], n_qubits)
