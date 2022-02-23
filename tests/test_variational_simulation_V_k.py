import numpy as np
from core.variational_simulation import V_k
import core.analytic as an

# n_qubits = 3
# J = 1/2
# B = 1/2
# fs = [[-1j*J, -1j*J, -1j*J], [-1j*B, -1j*B, -1j*B]]
# params = np.array([1.3, 1.0])
# ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# hs = [-J, -J, -J, -B, -B, -B]
# opsH = ["ZZI", "IZZ", "ZIZ", "XII", "IXI", "IIX"]
# k = 1
# vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
#                    0.35355339, -0.35355339, -0.35355339, -0.35355339]) + 1j*0
# vector = vector/np.linalg.norm(vector)
# v_k = V_k(params, fs, hs, ops, opsH, vector, k)
# v_k_test = an.V_k(params, fs, hs, ops, opsH, vector, k)
# print(v_k, v_k_test)

n_qubits = 2
J = 1/2
B = 1/2
fs = [[-1j*J], [-1j*B, -1j*B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]
hs = [-J, -B, -B]
opsH = ["ZZ", "XI", "IX"]
vector = np.array([0.5, 0.5, 0.5, 0.5]) +1j*0
vector = vector/np.linalg.norm(vector)
v_k = V_k(params, fs, hs, ops, opsH, vector, 1)
v_k_test = an.V_k(params, fs, hs, ops, opsH, vector, k)
print(v_k, v_k_test)
