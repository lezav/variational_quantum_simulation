import numpy as np
from core.variational_simulation import A_kqij
import core.analytic as an


# n_qubits = 3
# J = 1j*1/2
# B = 1j*1/2
# fs = [[-J, -J, -J], [-B, -B, -B]]
# params = np.array([1.0, 1.0])
# ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# k, q, i, j = 0, 1, 2, 0
# vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
#                    0.35355339, -0.35355339, -0.35355339, -0.35355339])+1j*0
# vector = vector/np.linalg.norm(vector)
# a_kqij = A_kqij(params, fs, ops, vector, k, q, i, j)
# a_kqij_test = an.A_kqij(params, fs, ops, vector, k, q, i, j)
# a_kqij, a_kqij_test

n_qubits = 2
J = 1j*1/2
B = 1j*1/2
fs = [[-J], [-B, -B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]
k, q, i, j = 1, 0, 0, 0
vector = np.array([0.5, 0.5, 0.5, 0.5]) +1j*0
vector = vector/np.linalg.norm(vector)
a_kqij = A_kqij(params, fs, ops, vector, k, q, i, j)
a_kqij_test = an.A_kqij(params, fs, ops, vector, k, q, i, j)
a_kqij, a_kqij_test
