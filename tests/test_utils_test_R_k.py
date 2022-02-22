import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.extensions import HamiltonianGate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from core.variational_simulation import R_k
from core.utils import test_R_k
import scipy


# create the circuit
n_qubits = 3
# define the parameters of the problem
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
operators = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
ZZI = np.kron(Z, np.kron(Z, I))
IZZ = np.kron(I, np.kron(Z, Z))
ZIZ = np.kron(Z, np.kron(I, Z))
param = np.array([1.0, 1.0])
# calculate the operation R_k
k = 0
R_test = test_R_k(param[k], fs[k], operators[k], n_qubits)
H_Z = fs[k][0]*ZZI + fs[k][1]*IZZ + fs[k][2]*ZIZ
scipy.linalg.expm(param[k]*H_Z) - R_test
