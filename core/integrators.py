#!/usr/bin/env python3

import numpy as np
import core.analytic as analytic_vqs
import core.variational_simulation as vqs

# Returns a rutine which takes the input parameters
# and solves the RHS of the ODE for d(theta)/dt
def define_ode(ops, h_ops, fs, hs, analytic=False, state=[]):
    def ode(theta):
        if analytic:
            # state = analytic.initial_state() # TODO
            A = analytic_vqs.A(theta, fs, ops, state)
            V = analytic_vqs.V(theta, fs, hs, ops, h_ops, state)
        else:
            n_qubits = len(fs[0])
            A = vqs.A(theta, fs, ops, n_qubits)
            V = vqs.V(theta, fs, hs, ops, h_ops, n_qubits)
        return np.linalg.solve(A, V)

    return ode                  # Closure with the relevant variables

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
