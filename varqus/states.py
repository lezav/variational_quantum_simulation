#!/usr/bin/env python3

import numpy as np
from varqus.analytic import R_k

def state_infidelity(state1, state2):
    s1 = state1 / np.linalg.norm(state1)
    s2 = state2 / np.linalg.norm(state2)
    fidelity = np.abs( np.vdot(s1, s2) )**2

    return 1 - fidelity

def state_from_parameters(params, ops, fs, initial_state):
    state = np.copy(initial_state)
    for l in range(len(params)):
        state = R_k(params[l], fs[l], ops[l]) @ state

    return state / np.linalg.norm(state)
