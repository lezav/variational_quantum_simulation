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
    Ops_k = fs_k[0]*Pauli(ops_k[0]).to_matrix()
    for j in range(1, n_k):
        Ops_k = fs_k[j]*Pauli(ops_k[j]).to_matrix()

    return scipy.linalg.expm(Ops_k)

def test_A_kqij(params, fs, ops, n_qubits, k, q, i, j):
    N = len(params)
    r_ki = []
    for m in range(k):
        r_ki.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
        print("unitaria", m)
    r_ki.append(Pauli(ops[k][i]).to_matrix())
    print("operador", k)
    for m in range(k, N):
        print("unitaria", m)
        r_ki.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
    r_qj = []
    for m in range(q):
        print("unitaria", m)
        r_qj.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
    r_qj.append(Pauli(ops[q][j]).to_matrix())
    print("operador", q)
    for m in range(q, N):
        print("unitaria", m)
        r_qj.append(test_R_k(params[m], fs[m], ops[m], n_qubits))
    R_ki = np.eye(2**n_qubits, 2**n_qubits) + 1j*0
    R_qj = np.eye(2**n_qubits, 2**n_qubits) + 1j*0
    for m in range(N+1):
        R_ki = r_ki[m]@R_ki
        R_qj = r_qj[m]@R_qj
    vector = np.array([ 0.35355339,  0.35355339, 0.35355339, -0.35355339,
                        0.35355339, -0.35355339, -0.35355339,
                        -0.35355339]).reshape(8, 1) +1j*0
    a_kqij = (vector.conj().T) @ (((R_ki.conj().T) @ R_qj) @ vector)
    a_kqij = np.conjugate(fs[k][i])*fs[q][j]*a_kqij
    a_kqij = (a_kqij + np.conjugate(a_kqij))/2
    return a_kqij
