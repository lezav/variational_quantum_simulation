import numpy as np
from core.variational_simulation import A_kq
import core.analytic as an

# n_qubits = 3
# J = 1j*1/2
# B = 1j*1/2
# fs = [[-J, -J, -J], [-B, -B, -B]]
# params = np.array([1.0, 1.0])
# ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# k, q= 0, 1
# vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
#                    0.35355339, -0.35355339, -0.35355339, -0.35355339])+1j*0
# vector = vector/np.linalg.norm(vector)
# a_kq = A_kq(params, fs, ops, vector, k, q)
# a_kq_test = an.A_kq(params, fs, ops, vector, k, q)
# a_kq, a_kq_test


n_qubits = 2
J = 1j*1/2
B = 1j*1/2
fs = [[-J], [-B, -B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]
k, q = 1, 0
vector = np.array([0.5, 0.5, 0.5, 0.5]) +1j*0
vector = vector/np.linalg.norm(vector)
a_kq = A_kq(params, fs, ops, vector, k, q)
a_kq_test = an.A_kq(params, fs, ops, vector, k, q)
a_kq, a_kq_test
