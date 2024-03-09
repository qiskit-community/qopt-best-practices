from qiskit_optimization.applications import Maxcut
from qiskit.circuit.library import QAOAAnsatz


def map_graph_to_qubo(graph):
    max_cut = Maxcut(graph)
    qp = max_cut.to_quadratic_program()
    return qp


def map_qubo_to_ising(qubo):
    qubitOp, offset = qubo.to_ising()
    return qubitOp, offset


def map_ising_to_circuit(hamiltonian, num_layers):
    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=num_layers)
    ansatz.measure_all()
    return ansatz
