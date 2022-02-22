#!/usr/bin/env python3

import numpy as np
from core.variational_simulation_with_v import V, A

# Returns a rutine which takes the input parameters
# and solves the RHS of the ODE for d(theta)/dt
def define_ode(ops, opsH, fs, hs, n_qubits):
    def ode(theta):
        # obtain M, V from theta, and the other arguments
        v = V(theta, fs, hs, ops, opsH, n_qubits)
        M = A(theta, fs, ops, n_qubits)

        return np.linalg.solve(M, v)

    return ode

# 1st-order integrator on dt
def euler(ode, x0, dt, Nt):

    acc = np.empty((Nt, len(x0)))
    acc[0, :] = x0
    for t in range(1, Nt):
        acc[t, :] = acc[t-1, :] + dt*ode(acc[t-1, :])

    return acc

# Arbitrary order integrator on dt.
# Requires 'order' evaluations of 'ode'
def rungekutta(ode, x0, dt, Nt, order = 4):
    acc = np.empty((Nt, len(x0)))
    acc[0, :] = x0
    for t in range(1, Nt):
        acc[t, :] = acc[t-1, :]
        for j in range(order, 0, -1):
            acc[t, :] = acc[t-1, :] + ode(acc[t, :]) * dt/j

    return acc
