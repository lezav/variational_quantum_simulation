import numpy as np
from core.variational_simulation import V
import core.analytic as an

n_qubits = 3
J = 1/2
B = 1/2
fs = [[-1j*J, -1j*J, -1j*J], [-1j*B, -1j*B, -1j*B]]
params = np.array([0.0, 0.0])
ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
hs = [-J, -J, -J, -B, -B, -B]
opsH = ["ZZI", "IZZ", "ZIZ", "XII", "IXI", "IIX"]
vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
                   0.35355339, -0.35355339, -0.35355339, -0.35355339]) + 1j*0
vector = vector/np.linalg.norm(vector)
v = V(params, fs, hs, ops, opsH, vector)
v_test = an.V(params, fs, hs, ops, opsH, vector)
print(v, v_test)

# n_qubits = 2
# J = 1/2
# B = 1/2
# fs = [[-1j*J], [-1j*B, -1j*B]]
# params = np.array([1.0, 1.0])
# ops = [["ZZ"], ["XI", "IX"]]
# hs = [-J, -B, -B]
# opsH = ["ZZ", "XI", "IX"]
# vector = np.array([0.5, 0.5, 0.5, 0.5]) +1j*0
# vector = vector/np.linalg.norm(vector)
# v = V(params, fs, hs, ops, opsH, vector)
# v_test = an.V(params, fs, hs, ops, opsH, vector)
# print(v, v_test)
