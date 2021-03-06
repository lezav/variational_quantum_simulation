# VarQuS - A package for Variational Quantum Simulation

## Description

The usual method to simulate quantum evolution in quantum systems is using the Trotter expansion. However, this usually requires circuits with many quantum operations and a large depth, which is not desirable in the current noisy era of quantum computation. An alternative is to use Variational Quantum Simulation, since the depth of the circuit is fixed during all the computation and we can choose variational forms that are efficient for different hardwares.

In this project, we implement a Python package that, given a Hamiltonian, a variational form and a state at t=0, gives us the state of the system for a future state at time t = T using a variational method. In the tutorial, we present the [theory of variational quantum simulation](https://doi.org/10.1103/PhysRevX.7.021050) and continue with the details of our implementation using Qiskit. Then, we show simulations for the evolution of quantum Isings models of 2 and 3 qubits and compare with experimental realizations using IBM's quantum processors and Qiskit Runtime. The results show an excellent agreement between theory and experiment.

## Installation

This package can be installed via pip by running
```sh
python -m pip install git+https://github.com/lezav/variational_quantum_simulation
```
