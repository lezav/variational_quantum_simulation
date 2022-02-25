#!/usr/bin/env python3

import numpy as np

# Gates which can be translated from strings to arrays
base_gates = {
    "I" : np.array([[1, 0], [0,  1]], dtype=complex),
    "X" : np.array([[0, 1], [1,  0]], dtype=complex),
    "Z" : np.array([[1, 0], [0, -1]], dtype=complex),
    "Y" : np.array([[0, -1j], [1j, 0]], dtype=complex)
}

# Translate a gate name to a matrix
def parse_gate(gates : str):
    U = np.array([1])
    for gate in gates:          # Iterate over subspaces
        U = np.kron(U, base_gates[gate])
    return U

# Get the hamiltonian as array from h_ops and hs
def get_hamiltonian(h_ops, hs, time=None):
    if callable(hs):
        assert time is not None, "If hs is callabe, you should provide the time to evaluate it"
        hs = hs(time)

    return sum( h * parse_gate(g) for (h, g) in zip(hs, h_ops) )

def infidelity(stateA, stateB):
    inf = []
    for i in range(len(stateA)):
        fid_i = np.abs(np.vdot(stateA[i], stateB[i]))**2.0
        inf_i = 1.0-fid_i
        inf.append(inf_i)
    return inf
