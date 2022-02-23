import numpy as np
import scipy.linalg as la

# Gates which can be translated from strings to arrays
base_gates = {
    "I" : np.array([[1, 0], [0,  1]], dtype=complex),
    "X" : np.array([[0, 1], [1,  0]], dtype=complex),
    "Z" : np.array([[1, 0], [0, -1]], dtype=complex),
}

def parse_gate(gates : str):
    U = np.array([1])
    for gate in gates:          # Iterate over subspaces
        U = np.kron(U, base_gates[gate])
    return U

def R_k(theta_k : float, fs_k : list, gates_k : list):
    """
    Calculate the unitary R_k.
    Args:
        theta_k:  float. theta_k parameter in the paper.
        fs_k: list. Contains the complex coefficients f_ki that appear in R_k.
        gates_k: list. Contains the operators sigma_ki that appear in R_k.
    Returns:
        R_k: Gate.
    """

    U = sum( f * parse_gate(g) for (f, g) in zip(fs_k, gates_k) )
    return la.expm(theta_k * U)

def A_kqij(theta, fs, gates, state, k, q, i, j):
    R_ki = np.copy(state)
    R_qj = np.copy(state)
    gate_ki = parse_gate(gates[k][i])
    gate_qj = parse_gate(gates[q][j])
    for l in range(len(theta)):
        if l == k:
            R_ki = gate_ki @ R_ki
        elif l == q:
            R_qj = gate_qj @ R_qj
        U = R_k(theta[l], fs[l], gates[l])
        R_ki = U @ R_ki
        R_qj = U @ R_qj

    coefs = 1j * np.conj(fs[k][i]) * fs[q][j]
    return 2*np.real( coefs * np.vdot(R_ki, R_qj) )

def A_kq(theta, fs, gates, state, k, q):
    s = 0.0
    for i in range(len(fs[k])):
        for j in range(len(fs[q])):
            s += A_kqij(theta, fs, gates, state, k, q, i, j)
    return s

def A(theta, fs, gates, state):
    N = len(theta)
    a = np.empty((N, N))
    for q in range(N):
        for k in range(q+1):    # Calculate only a half
            a[k, q] = A_kq(theta, fs, gates, state, k, q)

    return a - a.T              # Complete the other half

# TODO Could be combined with A_kqij to reuse R_ki
def V_kij(theta, fs, hs, gates, h_gates, state, k, i, j):
    R = np.copy(state)
    R_ki = np.copy(state)
    gate_ki = parse_gate(h_gates[k][i])
    for l in range(len(theta)):
        if l == k:
            R_ki = gate_ki @ R_ki
        U = R_k(theta[l], fs[l], gates[l])
        R = U @ R
        R_ki = U @ R_ki

    coefs = hs[j] * fs[k][i]
    return 2*np.real(coefs * np.vdot(R_ki, R))

def V_k(theta, fs, hs, gates, h_gates, state, k):
    s = 0.0
    for i in range(len(fs[k])):
        for j in range(len(hs)):
            s += V_kij(theta, fs, hs, gates, h_gates, state, k, i, j)

    return s

def V(theta, fs, hs, gates, h_gates, state):
    return np.array([ V_k(theta, fs, hs, gates, h_gates, state, k)
                      for k in range(len(theta)) ])
