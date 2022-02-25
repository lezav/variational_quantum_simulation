#!/usr/bin/env python3

import numpy as np

# 1st-order integrator on dt
def euler(ode, x0, dt, Nt):
    acc = np.empty_like(x0, shape=(Nt, len(x0)))
    acc[0, :] = x0
    for n in range(Nt-1):
        acc[n+1, :] = acc[n] + dt*ode(acc[n], n*dt)

    return acc

# 4-th order runge-kutta
def rk4(ode, x0, dt, Nt):
    acc = np.empty_like(x0, shape=(Nt, len(x0)))
    acc[0, :] = x0
    for n in range(Nt-1):
        tn = n*dt
        yn = acc[n, :]
        k1 = dt * ode(yn, tn)
        k2 = dt * ode(yn + k1/2, tn + dt/2)
        k3 = dt * ode(yn + k2/2, tn + dt/2)
        k4 = dt * ode(yn + k3,   tn + dt)
        acc[n+1, :] = yn + (k1 + 2*k2 + 2*k3 + k4)/6

    return acc
