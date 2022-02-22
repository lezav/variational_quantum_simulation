import numpy as np
from core.variational_simulation import A_kq

n_qubits = 3
J = 1j*1/2
B = 1j*1/2
fs = [[-J, -J, -J], [-B, -B, -B]]
params = np.array([1.0, 0.5])
ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]

A_kq(params, fs, ops, n_qubits, 0, 0)
