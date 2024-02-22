from qiskit import transpile

def transpile_abstract_circuit(circuit, backend):
    return transpile(circuit, backend=backend)