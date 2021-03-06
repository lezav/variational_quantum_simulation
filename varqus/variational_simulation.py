import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import HamiltonianGate
from varqus.utils import parse_gate
import scipy.linalg as la

# Default vqs backend is the Aer simulator
backend_simulator = Aer.get_backend('aer_simulator')


def A(params, fs, ops, vector, shots=2**13, backend=backend_simulator):
    """
    Calculate the matrix M from https://doi.org/10.1103/PhysRevX.7.021050
    Args:
        params:  array (N,). lambda_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        vector: array. Initial state of the system.
        shots: int. Number of shots that we want to simulate.
        backend: backend from Qiskit.
    Returns:
        a: array (N, N). Matrix M
    """
    N = params.shape[0]
    a = np.zeros((N, N))
    for q in range(N):
        for k in range(q+1):
            a[k, q] = A_kq(params, fs, ops, vector, k, q, shots, backend)
    a = a - a.T
    return a


def A_kq(params, fs, ops, vector, k, q, shots=2**13, backend=backend_simulator):
    """
    Calculate a term A_kq that appear in equation (12) of
    https://doi.org/10.1103/PhysRevX.7.021050
    Args:
        params:  array (N,). lambda_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        vector: array. Initial state of the system.
        k, q: int. params for which we want to calculate A_kq.
        shots: int. Number of shots that we want to simulate.
        backend: backend from Qiskit.
    Returns:
        a_qk: float. M_kQ in Eq. (12).
    """
    # select the elements from the lists
    n_k = len(fs[k])
    n_q = len(fs[q])
    a_kq = 0
    for i in range(n_k):
        for j in range(n_q):
            a_kq += A_kqij(params, fs, ops, vector, k, q, i, j, shots, backend)
    return a_kq


def A_kqij(params, fs, ops, vector, k, q, i, j, shots=16384, backend=backend_simulator):
    """
    Calculate A_kqij = i f*_ki f_qj <0|R^dagg_ki R_qj|0> that appear in Eq. (12).
    Args:
        params:  array (N,). lambda_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        k, q, i, j: int. params for which we want to calculate A_kqij.
        shots: int. Number of shots that we want to simulate.
        backend: backend from Qiskit.
    Returns:
        float. Dot product of derivatives given by Eq. (12).
    """
    # We create the circuit with n_qubits plus an ancilla.
    n_qubits = len(ops[0][0])
    qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
    qr_data = QuantumRegister(n_qubits, "data") # data register
    cr = ClassicalRegister(1, "cr") # classical register
    qc = QuantumCircuit(qr_ancilla, qr_data, cr)
    # preparate the ancilla in the state |0> + e^(theta)|1>
    a_kiqj = 2*np.abs(np.conjugate(1j*fs[k][i])*fs[q][j])
    theta_kiqj = np.angle(1j*np.conjugate(fs[k][i])*fs[q][j])
    qc.initialize(vector.flatten(), qr_data[:])
    qc.h(qr_ancilla)
    qc.p(theta_kiqj, qr_ancilla)
    qc.barrier()
    # Now we want to construct R_ki
    # apply R_1 ...R_k-1 gates
    for m in range(k):
        R = R_k(params[m], fs[m], ops[m])
        qc.append(R, qr_data[:])
    qc.barrier()
    # apply the controlled operation for sigma_ki
    qc.x(qr_ancilla)
    controlled_Uk = string2U(ops[k][i]).control(1)
    qc.append(controlled_Uk, qr_ancilla[:] + qr_data[::-1])
    qc.barrier()
    # apply R_k...R_N gates
    for m in range(k, q):
        R = R_k(params[m], fs[m], ops[m])
        qc.append(R, qr_data[:])
    qc.barrier()
    qc.x(qr_ancilla)
    # apply the controlled operation for sigma_qj
    controlled_Uq = string2U(ops[q][j]).control(1)
    qc.append(controlled_Uq, qr_ancilla[:] + qr_data[::-1])
    qc.barrier()
    # measure in the X basis with a number of shots
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla, cr)
    qc = transpile(qc, backend)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    #calculate a Re(e^(theta) <0|U|0>)
    Re_0U0 = (counts.get("0", 0) - counts.get("1", 0))/shots
    return a_kiqj*Re_0U0


def V(params, fs, hs, ops, opsH, vector, shots=2**13, backend=backend_simulator):
    """
    Calculate the matrix V from https://doi.org/10.1103/PhysRevX.7.021050
    Args:
        params:  array (N,). lambda_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        hs: list (N). Contains the complex coefficients h_j of the Hamiltonian.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        opsH: list (N). Contains the operators sigma_j.
        vector: array. Initial state of the system.
        shots: int. Number of shots that we want to simulate.
        backend: backend from Qiskit.
    Returns:
        v: array (N, ). Vector V
    """
    N = params.shape[0]
    v = np.zeros(N)
    for k in range(N):
        v[k] = V_k(params, fs, hs, ops, opsH, vector, k, shots, backend)
    return v


def V_k(params, fs, hs, ops, opsH, vector, k, shots=2**13, backend=backend_simulator):
    """
    Calculate a term V_k that appear in equation (13) of the paper.
    Args:
        params:  array (N,). lambda_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        hs: list (N). Contains the complex coefficients h_j of the Hamiltonian.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        opsH: list (N). Contains the operators sigma_j.
        vector: array. Initial state of the system.
        k: int. params for which we want to calculate V_k.
        shots: int. Number of shots that we want to simulate.
        backend: backend from Qiskit.
    Returns:
        v_k: float. V_k in Eq. (13).
    """
    n_k = len(fs[k])
    n_i = len(hs)
    v_k = 0.0
    for i in range(n_k):
        for j in range(n_i):
            v_k += V_kij(params, fs, hs, ops, opsH, vector, k, i, j, shots, backend)
    return v_k


def V_kij(params, fs, hs, ops, opsH, vector, k, i, j, shots=2**13, backend=backend_simulator):
    """
    Calculate V_kij = f*_ki h_j <0|R^dagg_ki sigma_J R|0>  in Eq. (13).
    Args:
        params:  array (N,). lambda_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        hs: list (N). Contains the complex coefficients h_j of the Hamiltonian.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        opsH: list (N). Contains the operators sigma_j.
        vector: array. Initial state of the system.
        k, i, j: int. params for which we want to calculate V_kij.
        shots: int. Number of shots that we want to simulate.
        backend: backend from Qiskit.
    Returns:
        v_kij: float. f*_ki h_j <0|R^dagg_ki sigma_J R|0> in Eq. 13.
    """
    # We create the circuit with n_qubits plus an ancilla.
    n_qubits = len(ops[0][0])
    qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
    qr_data = QuantumRegister(n_qubits, "data") # data register
    cr = ClassicalRegister(1, "cr") # classical register
    qc = QuantumCircuit(qr_ancilla, qr_data, cr)
    # preparate the ancilla in the state |0> + e^(theta)|1>
    N = params.shape[0]
    a_v_kij = 2*np.abs(np.conjugate(fs[k][i])*hs[j])
    theta_kij = np.angle(np.conjugate(fs[k][i])*hs[j])
    qc.initialize(vector.flatten(), qr_data[:])
    qc.h(qr_ancilla)
    qc.p(theta_kij, qr_ancilla)
    qc.barrier()
    # Now we want to construct R_ki
    # apply R_1 ...R_k-1 gates
    for m in range(k):
        R = R_k(params[m], fs[m], ops[m])
        qc.append(R, qr_data[:])
    qc.barrier()
    # apply the controlled operation for sigma_ki
    qc.x(qr_ancilla)
    controlled_Uk = string2U(ops[k][i]).control(num_ctrl_qubits=1)
    qc.append(controlled_Uk, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # apply R_k...R_N gates
    for m in range(k, N):
        R = R_k(params[m], fs[m], ops[m])
        qc.append(R, qr_data[:])
    qc.barrier()
    qc.x(qr_ancilla)
    # apply the controlled operation for sigma_j
    controlled_U = string2U(opsH[j]).control(num_ctrl_qubits=1)
    qc.append(controlled_U, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # measure in the X basis with a number of shots
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla, cr)
    qc = transpile(qc, backend)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    #calculate a Re(e^(theta) <0|U|0>)
    Re_0U0 = (counts.get("0", 0) - counts.get("1", 0))/shots
    return a_v_kij*Re_0U0


def string2U(op):
    """
    Given a string of n_qubits Paulis, calculate the corresponding gate
    Args:
        op: string. String of Paulis.
    Returns:
        gate: gate. Gate associated to op.

    """
    n_qubits = len(op)
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    for m in range(n_qubits):
        qc.unitary(Operator(parse_gate(op[m])), qr_data[m])
    gate = qc.to_gate(label=op)
    return gate


def R_k(params_k, fs_k, ops_k):
    """
    Calculate a unitary R_k(lambda_k) that we see in Eq. (7) from the paper
    https://doi.org/10.1103/PhysRevX.7.021050
    Args:
        params_k:  float. lambda_k parameter in the paper.
        fs_k: list. Contains the complex coefficients f_ki that appear in R_k.
        ops_k: list. Contains the operators sigma_ki that appear in R_k.
    Returns:
        r_k: Gate.
    """
    n_k = len(ops_k)
    Ops_k = fs_k[0]*Operator(parse_gate(ops_k[0]))
    for j in range(1, n_k):
        Ops_k += fs_k[j]*Operator(parse_gate(ops_k[j]))
    r_k = HamiltonianGate(1j*Ops_k , params_k, label="+".join(ops_k))
    return r_k
