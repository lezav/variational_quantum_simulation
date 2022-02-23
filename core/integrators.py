#!/usr/bin/env python3

import numpy as np

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
