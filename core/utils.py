#!/usr/bin/env python3

import numpy as np

# Gates which can be translated from strings to arrays
base_gates = {
    "I" : np.array([[1, 0], [0,  1]], dtype=complex),
    "X" : np.array([[0, 1], [1,  0]], dtype=complex),
    "Z" : np.array([[1, 0], [0, -1]], dtype=complex),
}

# Translate a gate name to a matrix
def parse_gate(gates : str):
    U = np.array([1])
    for gate in gates:          # Iterate over subspaces
        U = np.kron(U, base_gates[gate])
    return U
