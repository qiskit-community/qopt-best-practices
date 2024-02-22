from qiskit import transpile

def optimize_circuit(circuit, backend):
    return transpile(circuit, backend=backend)