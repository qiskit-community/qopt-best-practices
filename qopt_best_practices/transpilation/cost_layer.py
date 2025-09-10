"""Provides a method to obtain only the cost layer of QAOA."""

from qiskit import QuantumCircuit
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp


def get_cost_layer(cost_operator: SparsePauliOp):
    """Return the exponential of the cost operator only.

    Note: The cost_operator is allowed to contain parameters.
    """

    nqb = cost_operator.num_qubits

    init_state = QuantumCircuit(nqb)

    dummy_mixer_op = SparsePauliOp.from_sparse_list([("I", [i], 1) for i in range(nqb)], nqb)

    cost_layer = qaoa_ansatz(
        cost_operator,
        reps=1,
        initial_state=init_state,
        mixer_operator=dummy_mixer_op,
    )

    # Remove beta.
    cost_layer.assign_parameters({"β[0]": 0}, inplace=True)

    # The best practices rely on parameter naming conventions built-in qaoa_ansatz
    # Here we perform cheap validation checks to make sure these assumptions hold.
    # This prevents nasty surprises if `qaoa_anasatz` changes in Qiskit.
    if len(cost_layer.parameters) != len(cost_operator.parameters) + 1:
        raise ValueError("Incorrect number of parameters in cost layer.")

    if "γ[0]" not in [param.name for param in cost_layer.parameters]:
        raise ValueError("γ[0] is not in the parameters of the cost layer.")

    return cost_layer
