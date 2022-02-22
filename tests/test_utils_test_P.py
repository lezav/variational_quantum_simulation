import numpy as np
from core.utils import P
import scipy
from qiskit.quantum_info.operators import Operator, Pauli

ops = [["ZZI", "IZZ", "ZIZ"], ["XII", "IXI", "IIX"]]
Operator(Pauli("XII"))
P("XII")
