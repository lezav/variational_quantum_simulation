import numpy as np
import scipy.linalg as la
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit import Aer, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import HamiltonianGate
from qiskit.quantum_info import Statevector
from time import time, ctime

def main( backend, user_messenger, shots, dt, Nt ):

    t0 = ctime()
    t1 = time()

    J = 1/2
    B = 1/2
    fs = [[-1j*J,-1j*J,-1j*J], [-1j*B, -1j*B, -1j*B]]
    params_init = np.array([1.0, 1.0, 1.0])
    ops = [["ZZI","IZZ","ZIZ"], ["XII", "IXI", "IIX"]]
    hs = [-J, -J, -J, -B, -B, -B] # Hamiltonian non-dependent on time
    # hs = lambda t: [-2*J*np.cos(t), -B*np.cos(t), -B*np.sin(t)] # Hamiltonian parameters dependent on time # FAKE DATA FTW
    opsH = ["ZZI","IZZ","ZIZ","XII", "IXI", "IIX"]
    
    state = np.array([0.35355339,  0.35355339, 0.35355339, -0.35355339,
                    0.35355339, -0.35355339, -0.35355339, -0.35355339])+1j*0

    ode = define_vqs_ode(ops, opsH, fs, hs, state, analytic=False, shots=shots, backend=backend)    # Define the diff. equation RHS as a function of the parameters

    params_evolved = euler(ode, params_init, dt, Nt)              # Integrate in time!

    t2=time()

    return results_to_dict( params_evolved, shots, dt, Nt, t2-t1, backend.name(), t0 )

def results_to_dict(params_evolved, shots, dt, Nt, T, backend_name, T0 ):
    
    results_dict = {
        'params' : params_evolved ,
        'shots' : shots,
		'dt' : dt ,
		'Nt'  : Nt,
        'execution_time' : T,
        'date' : T0
        }

    return results_dict


######3

import numpy as np

# Returns a rutine which takes the input parameters
# and solves the RHS of the ODE for d(theta)/dt
def define_vqs_ode(ops, h_ops, fs, hs, state, analytic=False, shots=2**13, backend=Aer.get_backend('aer_simulator') ):
    def ode(theta, time):
        # Possibly time-dependent hamiltonian
        nonlocal hs             # Expect from closure
        if callable(hs):
            hs = hs(time)       # Resturn list of hamiltonian coefficients at 'time'

        A = A2(theta, fs, ops, state, shots, backend )
        V = V2(theta, fs, hs, ops, h_ops, state, shots, backend )

        return np.linalg.solve(A, V)

    return ode                  # Closure with the relevant variables

# Define an ODE for the Schrodinger equation
def define_schrodinger_ode(h_ops, hs):
    def schrodinger_ode(state, time=None):
        H = get_hamiltonian(h_ops, hs, time)
        return -1j * H @ state

    return schrodinger_ode

###############################

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit import Aer, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import *
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions import HamiltonianGate
from qiskit.quantum_info import Statevector

def initial_state(n_qubits):
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    qc.h(qr_data[:])
    for k in range(n_qubits-1):
        qc.cp(np.pi, k, k+1)
    qc.cp(np.pi, n_qubits-1, 0)
    return qc.to_gate(label="in_st")


def A2(params, fs, ops, vector, shots=2**13, backend=Aer.get_backend('aer_simulator')):
    """
    Calculate the matrix A
    """
    N = params.shape[0]
    a = np.zeros((N, N))
    for q in range(N):
        for k in range(q+1):
            a[k, q] = A_kq(params, fs, ops, vector, k, q, shots, backend )
    a = a - a.T
    return a


def A_kq(params, fs, ops, vector, k, q, shots=2**13, backend=Aer.get_backend('aer_simulator') ):
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
            a_kq += A_kqij(params, fs, ops, vector, k, q, i, j, shots, backend )
    return a_kq


def A_kqij(params, fs, ops, vector, k, q, i, j, shots=2**13, backend=Aer.get_backend('aer_simulator') ):
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
    n_qubits = len(ops[0][0])
    qr_ancilla = QuantumRegister(1, "ancilla") # ancilla register
    qr_data = QuantumRegister(n_qubits, "data") # data register
    cr = ClassicalRegister(1, "cr") # classical register
    qc = QuantumCircuit(qr_ancilla, qr_data, cr)
    # preparate the ancilla in the state |0> + e^(theta)|1>
    N = params.shape[0]
    a_kiqj = 2*np.abs(np.conjugate(1j*fs[k][i])*fs[q][j])
    theta_kiqj = np.angle(1j*np.conjugate(fs[k][i])*fs[q][j])
    # qc.append(initial_state(n_qubits), qr_data[:])
    qc.initialize(vector.flatten(), qr_data[:])
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
    controlled_Uk = string2U(ops[k][i], n_qubits).control(1)
    # controlled_Uk = controlled_gates(ops[k][i], k, i, n_qubits).control(1)
    qc.append(controlled_Uk, qr_ancilla[:] + qr_data[::-1])
    qc.barrier()
    # apply R_k...R_N gates
    for m in range(k, q):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    qc.x(qr_ancilla)
    # apply the controlled operation for sigma_qj
    controlled_Uq = string2U(ops[q][j], n_qubits).control(1)
    # controlled_Uq = controlled_gates(ops[q][j], q, j, n_qubits).control(1)
    qc.append(controlled_Uq, qr_ancilla[:] + qr_data[::-1])
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
    # print(qc.draw())
    # simulator = Aer.get_backend('aer_simulator')
    # simulator = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    # print(circ.draw())
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    #calculate a Re(e^(theta) <0|U|0>)
    Re_0U0 = (counts.get("0", 0) - counts.get("1", 0))/shots
    return a_kiqj*Re_0U0


def V2(params, fs, hs, ops, opsH, vector, shots=2**13, backend=Aer.get_backend('aer_simulator') ):
    """
    Calculate the matrix A
    """
    N = params.shape[0]
    v = np.zeros(N)
    for k in range(N):
        v[k] = V_k(params, fs, hs, ops, opsH, vector, k, shots, backend )
    return v


def V_k(params, fs, hs, ops, opsH, vector, k, shots=2**13, backend=Aer.get_backend('aer_simulator') ):
    """
    Calculate a term V_k that appear in equation (13) of the paper.
    Args:
        params:  array (N,). theta_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        k: int. params for which we want to calculate V_k.
    Returns:
        v_k: float. Eq. (13).
    """

    # select the elements from the lists
    n_k = len(fs[k])
    n_i = len(hs)
    v_k = 0
    for i in range(n_k):
        for j in range(n_i):
            v_k += V_kij(params, fs, hs, ops, opsH, vector, k, i, j, shots, backend )
    return v_k


def V_kij(params, fs, hs, ops, opsH, vector, k, i, j, shots=2**13, backend=Aer.get_backend('aer_simulator') ):
    """
    Calculate V_kij = f*_ki h_j <0|R^dagg_ki sigma_J R|0> that appear in Eq. (13).
    Args:
        params:  array (N,). theta_k parameters in the paper.
        fs: list (N). List that contain N lists. The inner lists contain the
            complex coefficients f_ki.
        hs: list (N). List that contain N lists. The inner lists contain the
            coefficients h_i.
        ops: list (N). List that contain N lists. The inner lists contain
            the operators sigma_ki.
        k, i, j: int. params for which we want to calculate V_kij.
    Returns:
        v_kij: float. Dot product of derivatives given by Eq. (10).
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
    # qc.append(initial_state(n_qubits), qr_data[:])
    qc.initialize(vector.flatten(), qr_data[:])
    qc.h(qr_ancilla)
    qc.p(theta_kij, qr_ancilla)
    qc.barrier()
    # Now we want to construct R_ki
    # apply R_1 ...R_k-1 gates
    for m in range(k):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    # apply the controlled operation for sigma_ki
    qc.x(qr_ancilla)
    controlled_Uk = string2U(ops[k][i], n_qubits).control(num_ctrl_qubits=1)
    qc.append(controlled_Uk, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # apply R_k...R_N gates
    for m in range(k, N):
        R = R_k(params[m], fs[m], ops[m], n_qubits)
        qc.append(R, qr_data[:])
    qc.barrier()
    qc.x(qr_ancilla)
    # apply the controlled operation for sigma_j
    controlled_U = string2U(opsH[j], n_qubits).control(num_ctrl_qubits=1)
    qc.append(controlled_U, qr_ancilla[:] + qr_data[:])
    qc.barrier()
    # measure in the X basis with a number of shots
    qc.h(qr_ancilla)
    qc.measure(qr_ancilla, cr)
    # print(qc.draw())
    # simulator = Aer.get_backend('aer_simulator')
    # simulator = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    # print(circ.draw())
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    #calculate a Re(e^(theta) <0|U|0>)
    Re_0U0 = (counts.get("0", 0) - counts.get("1", 0))/shots
    return a_v_kij*Re_0U0


def controlled_gates(ops_ki, k, i, n_qubits):
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    if int(k)==0:
        qc.z(qr_data[np.mod(i, 3)])
        qc.z(qr_data[np.mod(i + 1, 3)])
    else:
        qc.x(qr_data[i])
    return qc.to_gate(label=ops_ki)


def string2U(op, n_qubits):
    """
    Converts from string to gate.
    """
    qr_data = QuantumRegister(n_qubits, "data") # data register
    qc = QuantumCircuit(qr_data)
    for m in range(n_qubits):
        qc.unitary(Operator(parse_gate(op[m])), qr_data[m])
    # qc.unitary(Operator(parse_gate(op)), qr_data[::])
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
    Ops_k = fs_k[0]*Operator(parse_gate(ops_k[0]))
    for j in range(1, n_k):
        Ops_k += fs_k[j]*Operator(parse_gate(ops_k[j]))

    return HamiltonianGate(1j*Ops_k , params_k, label="+".join(ops_k))


#########################3


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

# Get the hamiltonian as array from h_ops and hs
def get_hamiltonian(h_ops, hs, time=None):
    if callable(hs):
        assert time is not None, "If hs is callabe, you should provide the time to evaluate it"
        hs = hs(time)

    return sum( h * parse_gate(g) for (h, g) in zip(hs, h_ops) )

######################3


# 1st-order integrator on dt
def euler(ode, x0, dt, Nt):
    acc = np.empty((Nt, len(x0)), dtype=type(x0[0]))
    acc[0, :] = x0
    for t in range(1, Nt):
        acc[t, :] = acc[t-1, :] + dt*ode(acc[t-1, :], dt*(t-1) )

    return acc

# 4-th order runge-kutta
def rk4(ode, x0, dt, Nt):
    acc = np.empty((Nt, len(x0)), dtype=type(x0[0]))
    acc[0, :] = x0
    for n in range(1, Nt):
        tn = (n-1)*dt
        yn = acc[n-1, :]
        k1 = dt * ode(yn, tn)
        k2 = dt * ode(yn + k1/2, tn + dt/2)
        k3 = dt * ode(yn + k2/2, tn + dt/2)
        k4 = dt * ode(yn + k3,   tn + dt)
        acc[n, :] = yn + (k1 + 2*k2 + 2*k3 + k4)/6

    return acc
