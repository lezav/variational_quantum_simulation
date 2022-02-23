import numpy as np
import scipy.linalg as la
# from qiskit.quantum_info.operators import Operator, Pauli
# import matplotlib.pyplot as plt

def ising_Hamiltonian(n_qubits):
    sz = np.array([ [1,0],[0,-1] ] )
    sx = np.array( [ [0,1], [1,0] ] )
    I = np.eye(2)
    if n_qubits ==2 :
        H_z = -J*2*np.kron(sz,sz)
        H_x = -B*(np.kron(sx,I)+ np.kron(I,sx))
    elif n_qubits == 3:
        H_z = -J*(np.kron(np.kron(sz,sz),I), np.kron(np.kron(I,sz),sz), np.kron(np.kron(sz,I),sz))
        H_x = -B*(np.kron(np.kron(sx,I),I)+ np.kron(np.kron(sx,I),I)+np.kron(I,np.kron(I,sx)))
    return H_z + H_x

def time_evolution(Hamiltonian, dt, Nt, img=True):
    if img==True:
        U = [la.expm(1j* dt*n *Hamiltonian) for n in range(Nt)]
    elif img==False:
        U = [la.expm(dt*n *Hamiltonian) for n in range(Nt)]
    return U

def state_evoluted(initial_state,Hamiltonian, dt, Nt, img=True):
    state_evoluted = time_evolution(Hamiltonian, dt, Nt, img=True) @ initial_state
    return state_evoluted
