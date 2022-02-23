import numpy as np
from core.variational_simulation_with_v import A, V

n_qubits = 2
J = 1/2
B = 1/2
fs = [[-1j*J], [-1j*B, -1j*B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]

hs = [-2.0*J, -B, -B]
opsH = ["ZZ", "XI", "IX"]

# a = A(params, fs, ops, n_qubits)
v = V(params, fs, hs, ops, opsH, n_qubits)

# print(a)
print(v)
