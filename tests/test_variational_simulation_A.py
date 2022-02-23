import numpy as np
from core.variational_simulation import A
import core.analytic as an

n_qubits = 3
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([0.0, 0.0])
ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
                   0.35355339, -0.35355339, -0.35355339, -0.35355339])+1j*0
vector = vector/np.linalg.norm(vector)
a = A(params, fs, ops, vector)
test_a = an.A(params, fs, ops, vector)
a, test_a

# n_qubits = 2
# J = 1j*1/2
# B = 1j*1/2
# fs = [[-J], [-B, -B]]
# params = np.array([0.0, 0.0])
# ops = [["ZZ"], ["XI", "IX"]]
# vector = np.array([0.5, 0.5, 0.5, 0.5]) +1j*0
# vector = vector/np.linalg.norm(vector)
# a = A(params, fs, ops, vector)
# test_a = an.A(params, fs, ops, vector)
# a, test_a
