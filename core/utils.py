import numpy as np
import scipy
from qiskit.quantum_info.operators import Operator, Pauli


def test_R_k(params_k, fs_k, ops_k, n_qubits):
    """
    Calculate the unitary R_k.
    Args:
        params_k:  float. theta_k parameter in the paper.
        fs_k: list. Contains the complex coefficients f_ki that appear in R_k.
        ops_k: list. Contains the operators sigma_ki that appear in R_k.
    Returns:
        R: Gate.
    """

    n_k = len(ops_k)
    Ops_k = fs_k[0]*P(ops_k[0])
    for j in range(1, n_k):
        Ops_k += fs_k[j]*P(ops_k[j])

    return scipy.linalg.expm(params_k*Ops_k)

def test_A_kqij(params, fs, ops, n_qubits, k, q, i, j, vector):
    N = len(params)
    r_ki = []
    for m in range(k):
        r_ki.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
        # print("unitaria", m)
    r_ki.append(P(ops[k][i]))
    # print("operador", k)
    for m in range(k, N):
        # print("unitaria", m)
        r_ki.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
    r_qj = []
    for m in range(q):
        # print("unitaria", m)
        r_qj.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
    r_qj.append(P(ops[q][j]))
    # print("operador", q)
    for m in range(q, N):
        # print("unitaria", m)
        r_qj.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
    R_ki = np.eye(2**n_qubits, 2**n_qubits) + 1j*0
    R_qj = np.eye(2**n_qubits, 2**n_qubits) + 1j*0
    for m in range(N+1):
        R_ki = r_ki[m]@R_ki
        R_qj = r_qj[m]@R_qj
    a_kqij = (vector.conj().T) @ (((R_ki.conj().T) @ R_qj) @ vector)
    a_kqij = 1j*np.conjugate(fs[k][i])*fs[q][j]*a_kqij
    a_kqij = np.real(a_kqij + np.conjugate(a_kqij))
    return a_kqij


def test_A_kq(params, fs, ops, n_qubits, k, q, vector):
    n_k = len(fs[k])
    n_q = len(fs[q])
    a_kq = 0
    for i in range(n_k):
        for j in range(n_q):
            a_kq += test_A_kqij(params, fs, ops, n_qubits, k, q, i, j, vector)
    return a_kq


def test_A(params, fs, ops, n_qubits, vector):
    """
    Calculate the matrix A
    """
    N = params.shape[0]
    a = np.zeros((N, N))
    for q in range(N):
        for k in range(q+1):
            a[k, q] = test_A_kq(params, fs, ops, n_qubits, k, q, vector)
    a = a - a.T
    return a


def P(s):
    d = {"I":np.array([[1, 0], [0, 1]]) +1j*0,
         "X":np.array([[0, 1], [1, 0]])+1j*0,
         "Z":np.array([[1,  0], [0, -1]])+1j*0}
    p = np.array([1])
    for st in s:
        # print(st)
        p = np.kron(p, d[st])
    return p
