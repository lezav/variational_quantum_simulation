import pennylane as qml
from pennylane import numpy as np


def string2op(s, wires):
    d = {"X":qml.PauliX(wires=wires),
         "Y":qml.PauliY(wires=wires),
         "Z":qml.PauliZ(wires=wires),
         "I":qml.Identity(wires=wires)}
    return d[s]

def string2gate(s, params, wires):
    d = {"X":qml.RX(params, wires=wires),
         "Y":qml.RY(params, wires=wires),
         "Z":qml.RZ(params, wires=wires),
         "I":qml.Identity(wires=wires)}
    return d[s]
