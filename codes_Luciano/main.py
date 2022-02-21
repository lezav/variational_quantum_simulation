from numbers import Complex
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector

def append_gate( qc, gate, qubits, cqubits=None, angle=0 ) :

    if gate=='ry':
        qc.ry(angle,qubits)
    if gate=='cx':
        qc.cx( qubits[0], qubits[1])
    if gate=='cy':
        qc.cy( cqubits, qubits)
    if gate=='cxx':
        qc.cx( cqubits, qubits[0])
        qc.cx( cqubits, qubits[1])
    if gate=='cyy':
        qc.cy( cqubits, qubits[0])
        qc.cy( cqubits, qubits[1])


def circuits( R, Uk, Uq, n_params ):

    circ = []

    params_circuit = ParameterVector( "p", n_params )

    n_gates = len(R)
    n_gates_paulis = len(Uq)

    for k in range(n_gates):
        for q in range( k, n_gates_paulis):
            
            if Uk[k] is not None and Uq[q] is not None :

                qc = QuantumCircuit( 3, 1 )
                qc.h(2)
                qc.barrier()
                idx_par = 0

                for j in range(n_gates):
                    
                    if j == k :
                        qc.x(2)
                        append_gate( qc, Uk[k][0], Uk[k][1], cqubits=2 )
                        qc.x(2)
                    
                    if j == q or ( j==n_gates-1 and q>n_gates-1 ) :
                        append_gate( qc, Uq[q][0], Uq[q][1], cqubits=2  )
                    
                    append_gate( qc, R[j][0], R[j][1], angle=params_circuit[idx_par] )
                    
                    if R[j][0] != 'cx':
                        idx_par += 1

                    qc.barrier()

                qc.h(2)
                qc.measure(2,0)
                circ.append( qc )

    return circ


def Euler_step( results, shots, n_params, n_paulis  ):

    counts = results.get_counts()

    ExpVal = []
    for count in counts:
        temp = 0
        for p in count:
            temp += count[p]*(-1)**int(p) / shots
        ExpVal.append( temp ) 
    ExpVal

    A = np.zeros( [n_params, n_params])
    C = np.zeros( n_params)

    idx = 0
    for k in range(n_params):
        for q in range( k, n_params+n_paulis):
            if q < n_params :
                A[k,q] = ExpVal[idx] /4. 
                A[q,k] = A[k,q]
            else:
                C[k] = ExpVal[idx] / 2. 
            idx += 1
    dt = np.linalg.solve( A, C )
    print(A, C)
    return dt



def VariationalSimulation( R, Uq, Uk, params, steps, shots ):

    simulator = Aer.get_backend('aer_simulator')

    n_params = len(params)
    circ = circuits( R, Uk, Uq, n_params )

    params_evolved = []
    params_evolved.append( params.copy() )

    for _ in range(steps):

        circ_par = [ qc.assign_parameters(params) for qc in circ  ]
        job = simulator.run( circ_par , shots=shots )
        dparams = Euler_step( job.result(), shots, n_params, 1 )
        params += dparams
        params_evolved.append( params.copy() )

    return params_evolved




