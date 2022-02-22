import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit import Aer, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import HamiltonianGate
from qiskit.quantum_info import Statevector
from core.utils import P

def initial_state(n_qubits):
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    qc.h(qr_data[:])
    for k in range(n_qubits-1):
        qc.cp(np.pi, k, k+1)
    qc.cp(np.pi, k + 1, 0)
    return qc.to_gate(label="in_st")


def A(params, fs, ops, n_qubits):
    """
    Calculate the matrix A
    """
    N = params.shape[0]
    a = np.zeros((N, N))
    for q in range(N):
        for k in range(q+1):
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


def A_kqij_3qubits(params, fs, ops, n_qubits, k, q, i, j, shots=8192):
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
    a_kiqj = 2*np.abs(np.conjugate(1j*fs[k][i])*fs[q][j])
    theta_kiqj = np.angle(1j*np.conjugate(fs[k][i])*fs[q][j])
    qc.append(initial_state(n_qubits), qr_data[:])
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
    # controlled_Uk = string2U(ops[k][i], n_qubits).control(1)
    controlled_Uk = controlled_gates(ops[k][i], k, i, n_qubits).control(1)
    qc.append(controlled_Uk, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # apply R_k...R_N gates
    for m in range(k, q):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    qc.x(qr_ancilla)
    # apply the controlled operation for sigma_qj
    # controlled_Uq = string2U(ops[q][j], n_qubits).control(1)
    controlled_Uq = controlled_gates(ops[q][j], q, j, n_qubits).control(1)
    qc.append(controlled_Uq, qr_ancilla[:] + qr_data[:])
    # qc.cx(qr_ancilla, qr_data[0])
    qc.barrier()
    # apply the operations R_q ...R_N
    # for m in range(q, N):
    #     R = R_k(params[m], fs[m], ops[m], n_qubits)
    #     qc.append(R, qr_data[:])
    # qc.barrier()
    # measure in the X basis with a number of shots
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla, cr)
    print(qc.draw())
    simulator = Aer.get_backend('aer_simulator')
    # simulator = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, simulator)
    # print(circ.draw())
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    #calculate a Re(e^(theta) <0|U|0>)
    Re_0U0 = (counts.get("0", 0) - counts.get("1", 0))/shots
    return a_kiqj*Re_0U0


def controlled_gates(ops, k, i, n_qubits):
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    if int(k)==0:
        qc.z(qr_data[np.mod(i, 3)])
        qc.z(qr_data[np.mod(i + 1, 3)])
    else:
        qc.x(qr_data[i])
    return qc.to_gate(label=ops)


def string2U(op, n_qubits):
    """
    Converts from string to gate.
    """
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    qc.unitary(Operator(P(op)), qr_data[::-1])
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

    n_k = len(ops_k)
    Ops_k = fs_k[0]*Operator(P(ops_k[0]))
    for j in range(1, n_k):
        Ops_k += fs_k[j]*Operator(P(ops_k[j]))

    return HamiltonianGate(1j*Ops_k , params_k, label="+".join(ops_k))
