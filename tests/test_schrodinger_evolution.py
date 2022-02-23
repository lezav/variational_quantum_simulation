import numpy as np
from core.variational_simulation import V
import core.analytic as an
from core.schrodinger import time_evolution, state_evoluted
from core.integrators import euler
from core.ode import define_vqs_ode, define_schrodinger_ode
# n_qubits = 3
# J = 1/2
# B = 1/2
# fs = [[-1j*J, -1j*J, -1j*J], [-1j*B, -1j*B, -1j*B]]
# params = np.array([0.0, 0.0])
# ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
# hs = [-J, -J, -J, -B, -B, -B]
# opsH = ["ZZI", "IZZ", "ZIZ", "XII", "IXI", "IIX"]
# vector = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
#                    0.35355339, -0.35355339, -0.35355339, -0.35355339]) + 1j*0
# vector = vector/np.linalg.norm(vector)
# dt = 0.01
# Nt = 50
# U = time_evolution(opsH, hs, dt, Nt, img=True)
# state = state_evoluted(vector, opsH, hs, dt, Nt)


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
#Schrodinger
dt = 0.01
Nt = 50
vector_ev = state_evoluted(vector, opsH, hs, dt, Nt)
#Variational
state = state_evoluted(state_evoluted(vector, ["ZZ"], [-J], 1, 2)[:, 1],
                ["XI", "IX"], [-B, -B], 1, 2)[:, 1]
ode = define_vqs_ode(ops, opsH, fs, hs, analytic=True, state = state)
params_evolved = euler(ode, params, dt, Nt)
params_evolved[-1, :]
# state_evolved = state_evoluted(state_evoluted(state, ["ZZ"], [-J], params_evolved[-1, 0], 2)[:, 1],
#                 ["XI", "IX"], [-B, -B], params_evolved[-1, 1], 2)[:, 1]
# np.abs(np.dot(state_evolved, vector_ev[:, -1]))**2
schrodinger = define_schrodinger_ode(opsH, hs)
state_evolved = euler(schrodinger, state, dt, Nt)
