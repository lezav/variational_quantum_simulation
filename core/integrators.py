#!/usr/bin/env python3

import numpy as np


# TODO After the interface is complete,
# rplace the 'many arguments' here
def define_ode(many, arguments):
    def ode(theta):
        # obtain M, V from theta, and the other arguments
        M, V = # ...

        return np.linalg.solve(M, V)

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
