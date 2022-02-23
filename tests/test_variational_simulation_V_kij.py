import numpy as np
from core.variational_simulation import A_kqij, V_kij
import core.analytic as an
from core.utils_with_v_test import test_V_kij

# n_qubits = 3
# J = 1/2
# B = 1/2
# fs = [[-1j*J, -1j*J, -1j*J], [-1j*B, -1j*B, -1j*B]]
# params = np.array([1.3, 1.0])
# ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# hs = [-J, -J, -J, -B, -B, -B]
# opsH = ["ZZI", "IZZ", "ZIZ", "XII", "IXI", "IIX"]
# k, i, j = 0, 1, 2
# vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
#                    0.35355339, -0.35355339, -0.35355339, -0.35355339])+1j*0
# vector = vector/np.linalg.norm(vector)
# v_kij = V_kij(params, fs, hs, ops, opsH, vector, k, i, j)
# v_kij_test = an.V_kij(params, fs, hs, ops, opsH, vector, k, i, j)
# v_kij_test_vv = test_V_kij(params, fs, hs, ops, opsH, vector, n_qubits, k, i, j)
# print(v_kij, v_kij_test, v_kij_test_vv)

n_qubits = 2
J = 1/2
B = 1/2
fs = [[-1j*J], [-1j*B, -1j*B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]
hs = [-J, -B, -B]
opsH = ["ZZ", "XI", "IX"]
k, i, j = 1, 1, 1
vector = np.array([0.5, 0.5, 0.5, 0.5]) +1j*0
vector = vector/np.linalg.norm(vector)
v_kij = V_kij(params, fs, hs, ops, opsH, vector, k, i, j)
v_kij_test = an.V_kij(params, fs, hs, ops, opsH, vector, k, i, j)
v_kij_test_vv = test_V_kij(params, fs, hs, ops, opsH, vector, n_qubits, k, i, j)
print(v_kij, v_kij_test, v_kij_test_vv)
