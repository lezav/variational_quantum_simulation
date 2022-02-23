#!/usr/bin/env python3

import numpy as np
import core.analytic as analytic_vqs
import core.variational_simulation as vqs
from core.utils import get_hamiltonian

# Returns a rutine which takes the input parameters
# and solves the RHS of the ODE for d(theta)/dt
def define_vqs_ode(ops, h_ops, fs, hs, analytic=False, state=[]):
    def ode(theta, time):
        # Possibly time-dependent hamiltonian
        nonlocal hs             # Expect from closure
        if callable(hs):
            hs = hs(time)       # Resturn list of hamiltonian coefficients at 'time'

        if analytic:
            # state = analytic.initial_state() # TODO
            A = analytic_vqs.A(theta, fs, ops, state)
            V = analytic_vqs.V(theta, fs, hs, ops, h_ops, state)
        else:
            n_qubits = len(h_ops[0])
            A = vqs.A(theta, fs, ops, n_qubits)
            V = vqs.V(theta, fs, hs, ops, h_ops, n_qubits)
        return np.linalg.solve(A, V)

    return ode                  # Closure with the relevant variables

# Define an ODE for the Schrodinger equation
def define_schrodinger_ode(h_ops, hs):
    def schrodinger_ode(state, time=None):
        H = get_hamiltonian(h_ops, hs, time)
        return -1j * H @ state

    return schrodinger_ode

# 1st-order integrator on dt
def euler(ode, x0, dt, Nt):
    acc = np.empty((Nt, len(x0)), dtype=type(x0[0]))
    acc[0, :] = x0
    for t in range(1, Nt):
        acc[t, :] = acc[t-1, :] + dt*ode(acc[t-1, :], dt*(t-1) )

    return acc

# 4-th order runge-kutta
def rk4(ode, x0, dt, Nt):
    acc = np.empty((Nt, len(x0)), dtype=type(x0[0]))
    acc[0, :] = x0
    for n in range(1, Nt):
        tn = (n-1)*dt
        yn = acc[n-1, :]
        k1 = dt * ode(yn, tn)
        k2 = dt * ode(yn + k1/2, tn + dt/2)
        k3 = dt * ode(yn + k2/2, tn + dt/2)
        k4 = dt * ode(yn + k3,   tn + dt)
        acc[n, :] = yn + (k1 + 2*k2 + 2*k3 + k4)/6

    return acc
