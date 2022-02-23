import numpy as np
from core.variational_simulation import A_kq
from core.utils import test_A_kq

# n_qubits = 3
# J = 1j*1/2
# B = 1j*1/2
# fs = [[-J, -J, -J], [-B, -B, -B]]
# params = np.array([1.0, 1.0])
# ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# vector = np.array([ 0.35355339,  0.35355339,
#         0.35355339, -0.35355339,
#         0.35355339, -0.35355339,
#        -0.35355339, -0.35355339]).reshape(8, 1) +1j*0
# k, q = 0, 1
# a_kq = A_kq(params, fs, ops, n_qubits, k, q)
# a_kq_test = test_A_kq(params, fs, ops, n_qubits, k, q, vector)
# a_kq, a_kq_test


n_qubits = 2
J = 1j*1/2
B = 1j*1/2
fs = [[-J], [-B, -B]]
params = np.array([1.0, 1.0])
ops = [["ZZ"], ["XI", "IX"]]
k, q = 0, 1
vector = np.array([0.5, 0.5, 0.5, 0.5]).reshape(4, 1) +1j*0
a_kq = A_kq(params, fs, ops, n_qubits, k, q)
a_kq_test = test_A_kq(params, fs, ops, n_qubits, k, q, vector)
a_kq, a_kq_test
