import pennylane as qml
from pennylane import numpy as np
from core.utils_pennylane import string2op, string2gate

dev = qml.device("default.qubit", wires=["a", 0, 1, 2])
@qml.qnode(dev)
def A_kqij(params, fs, ops, n_qubits, k, q, i, j):
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

    N = params.shape[0]
    a_kiqj = 2*np.abs(1j*np.conjugate(fs[k][i])*fs[q][j])
    theta_kiqj = np.angle(1j*np.conjugate(fs[k][i])*fs[q][j])

    qml.Hadamard(wires="a")
    qml.PhaseShift(theta_kiqj, wires="a")
    # Now we want to construct R_ki
    # apply R_1 ...R_k-1 gates
    for m in range(k):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
    # apply the controlled operation for sigma_ki
    qml.PauliX(wires="a")
    U = qml.transforms.get_unitary_matrix(ops_ki)(ops[k][i], n_qubits)
    qml.ControlledQubitUnitary(U, control_wires=["a"], wires=range(n_qubits),
                               control_values='1')
    # controlled_Uk = string2U(ops[k][i], n_qubits).control(num_ctrl_qubits=1)
    # qc.append(controlled_Uk, qr_ancilla[:] + qr_data[:])
    # qc.barrier()
    # # apply R_k...R_N gates
    # for m in range(k, q):
    #     R = R_k(params[m], fs[m], ops[m], n_qubits)
    #     qc.append(R, qr_data[:])
    # qc.barrier()
    # qc.x(qr_ancilla)
    # # apply the controlled operation for sigma_qj
    # controlled_Uq = string2U(ops[q][j], n_qubits).control(num_ctrl_qubits=1)
    # qc.append(controlled_Uq, qr_ancilla[:] + qr_data[:])
    # qc.barrier()
    # # apply the operations R_q ...R_1
    # for m in range(q, N):
    #     R = R_k(params[m], fs[m], ops[m], n_qubits)
    #     qc.append(R, qr_data[:])
    # qc.barrier()
    # # measure in the X basis with a number of shots
    # qc.h(qr_ancilla)
    # qc.measure(qr_ancilla, cr)
    # print(qc.draw())
    # simulator = Aer.get_backend('aer_simulator')
    # # simulator = Aer.get_backend('statevector_simulator')
    # qc = transpile(qc, simulator)
    # # print(circ.draw())
    # result = simulator.run(qc, shots=shots).result()
    # counts = result.get_counts(qc)
    # #calculate a Re(e^(theta) <0|U|0>)
    # Re_0U0 = (2*counts["0"]/shots - 1)
    return qml.expval(qml.PauliX(wires="a"))


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
    for m in range(n_k):
        for n in range(n_qubits):
             string2gate(ops_k[m][n], params_k, wires=n)
             # print(string2gate(ops_k[m][n], params_k, wires=n))


def ops_ki(op, n_qubits):

    for m in range(n_qubits):
        string2op(op[m], m)
        # print(string2op(op[m], m))
