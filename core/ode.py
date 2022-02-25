#!/usr/bin/env python3

import numpy as np
import core.analytic as analytic_vqs
import core.variational_simulation as vqs
from core.utils import get_hamiltonian

# Returns a rutine which takes the input parameters
# and solves the RHS of the ODE for d(theta)/dt
def define_vqs_ode(ops, h_ops, fs, hs, state, analytic=False, shots=None, backend=vqs.backend_simulator):
    def ode(theta, time):
        # Possibly time-dependent hamiltonian
        nonlocal hs             # Expect from closure
        if callable(hs):
            hs = hs(time)       # Resturn list of hamiltonian coefficients at 'time'

        if analytic:
            A = analytic_vqs.A(theta, fs, ops, state)
            V = analytic_vqs.V(theta, fs, hs, ops, h_ops, state)
        else:
            assert shots is not None, "To run in a circuit, you should provide the number of shots"
            A = vqs.A(theta, fs, ops, state, shots, backend)
            V = vqs.V(theta, fs, hs, ops, h_ops, state, shots, backend)
        return np.linalg.solve(A, V)

    return ode                  # Closure with the relevant variables

# Define an ODE for the Schrodinger equation
def define_schrodinger_ode(h_ops, hs):
    def schrodinger_ode(state, time=None):
        H = get_hamiltonian(h_ops, hs, time)
        return -1j * H @ state

    return schrodinger_ode
