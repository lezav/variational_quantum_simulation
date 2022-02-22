import numpy as np
from core.variational_simulation import A_kq, V_k

n_qubits = 2
J = 1/2
B = 1/2
fs = [[-1j*J], [-1j*B, -1j*B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]

hs = [-2.0*J, -B, -B]
opsH = ["ZZ", "XI", "IX"]

# a_kq = A_kq(params, fs, ops, n_qubits, 0, 0)


v_k = V_k(params, fs, hs, ops, opsH, n_qubits, 1)

print(v_k)