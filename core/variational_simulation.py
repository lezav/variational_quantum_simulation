import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit import Aer, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import HamiltonianGate


def A(params, fs, ops, n_qubits):
    """
    Calculate the matrix A
    """
    N = params.shape[0]
    a = np.zeros((N, N))
    for k in range(N):
        for q in range(N):
            a[k, q] = A_kq(params, fs, ops, n_qubits, k, q)
    return a


def A_kq(params, fs, ops, n_qubits, k, q):
    """
    Calculate a term A_kq that appear in equation (21) of the paper.
    Args:
        params:  array (N,). theta_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        k, q: int. params for which we want to calculate A_kq.
    Returns:
        a_qk: float. Eq. (21).
    """

    # select the elements from the lists
    n_k = len(fs[k])
    n_q = len(fs[q])
    a_kq = 0
    for i in range(n_k):
        for j in range(n_q):
            a_kq += A_kqij(params, fs, ops, n_qubits, k, q, i, j)
    return a_kq


def A_kqij(params, fs, ops, n_qubits, k, q, i, j, shots=8192):
    """
    Calculate A_kqij = f*_ki f_qj <0|R^dagg_ki R_qj|0> that appear in Eq. (21).
    Args:
        params:  array (N,). theta_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        k, q, i, j: int. params for which we want to calculate A_kqij.
    Returns:
        a_ij: float. Dot product of derivatives given by Eq. (10).
    """
    # We create the circuit with n_qubits plus an ancilla.
    qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
    qr_data = QuantumRegister(n_qubits, "data") # data register
    cr = ClassicalRegister(1, "cr") # classical register
    qc = QuantumCircuit(qr_ancilla, qr_data, cr)
    # preparate the ancilla in the state |0> + e^(theta)|1>
    N = params.shape[0]
    a_kiqj = 2*np.abs(1j*np.conjugate(fs[k][i])*fs[q][j])
    theta_kiqj = np.angle(1j*np.conjugate(fs[k][i])*fs[q][j])
    qc.h(qr_ancilla)
    qc.p(theta_kiqj, qr_ancilla)
    qc.barrier()
    # Now we want to construct R_ki
    # apply R_1 ...R_k-1 gates
    for m in range(k):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    # apply the controlled operation for sigma_ki
    qc.x(qr_ancilla)
    controlled_U = string2U(ops[k][i], n_qubits).control(num_ctrl_qubits=1)
    qc.append(controlled_U, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # apply R_k...R_N gates
    for m in range(k, N):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    # Now we want to construct R_qj
    # apply R_N ...R_q+1
    for m in range(N-1, q, -1):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    qc.x(qr_ancilla)
    # apply the controlled operation for sigma_qj
    controlled_U = string2U(ops[q][k], n_qubits).control(num_ctrl_qubits=1)
    qc.append(controlled_U, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # apply the operations R_q ...R_1
    for m in range(q, -1, -1):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    # measure in the X basis with a number of shots
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla, cr)
    # print(qc.draw())
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(qc, simulator)
    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    # calculate a Re(e^(theta) <0|U|0>)
    Re_0U0 = (counts["0"] - counts["1"])/shots
    return a_kiqj*Re_0U0


def string2U(op, n_qubits):
    """
    Converts from string to gate.
    """
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    qc.append(Operator(Pauli(op)), qr_data[:])
    return qc.to_gate(label=op)


def R_k(params_k, fs_k, ops_k, n_qubits):
    """
    Calculate the unitary R_k.
    Args:
        params_k:  float. theta_k parameter in the paper.
        fs_k: list. Contains the complex coefficients f_ki that appear in R_k.
        ops_k: list. Contains the operators sigma_ki that appear in R_k.
    Returns:
        R: Gate.
    """
    # qr_data = QuantumRegister(n_qubits, "data") # data register
    # qc = QuantumCircuit(qr_data)
    n_k = len(ops_k)
    Ops_k = fs_k[0]*Operator(Pauli(ops_k[0]))
    for j in range(1, n_k):
        Ops_k = fs_k[j]*Operator(Pauli(ops_k[j]))

    return HamiltonianGate(1j*Ops_k , params_k, label="+".join(ops_k))
