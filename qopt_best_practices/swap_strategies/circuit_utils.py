"""Circuit utils"""

from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)

from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import ParameterVector

def make_meas_map(circuit: QuantumCircuit) -> dict:
    """Return a mapping from qubit index (the key) to classical bit (the value).

    This allows us to account for the swapping order.
    """
    creg = circuit.cregs[0]
    qreg = circuit.qregs[0]

    meas_map = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            meas_map[qreg.index(inst.qubits[0])] = creg.index(inst.clbits[0])

    return meas_map

def apply_swap_strategy(circuit, swap_strategy, edge_coloring):

    pm_pre = PassManager(
        [
            FindCommutingPauliEvolutions(),
            Commuting2qGateRouter(
                swap_strategy,
                edge_coloring,
            )
        ]
    )
    return pm_pre.run(circuit)

def apply_qaoa_layers(circuit, meas_map, num_layers, gamma=None, beta=None):

    num_qubits = circuit.num_qubits
    new_circuit = QuantumCircuit(num_qubits, num_qubits)

    if gamma is None or beta is None:

        gamma = ParameterVector("γ", num_layers)
        beta = ParameterVector("β", num_layers)

    for layer in range(num_layers):
        bind_dict = {circuit.parameters[0]: gamma[layer]}
        bound_hc = circuit.assign_parameters(bind_dict)
        if layer % 2 == 0:
           new_circuit.append(bound_hc, range(num_qubits))
        else:
            new_circuit.append(bound_hc.reverse_ops(), range(num_qubits))

        new_circuit.rx(-2 * beta[layer], range(num_qubits))

    for qidx, cidx in meas_map.items():
        new_circuit.measure(qidx, cidx)

    return new_circuit

def create_qaoa_swap_circuit(
    cost_operator: list[tuple[str, float]],
    swap_strategy: SwapStrategy,
    edge_coloring: dict = None,
    theta: list[float] = None,
    qaoa_layers: int = 1,
    mixer_operator: QuantumCircuit = None,
    initial_state: QuantumCircuit = None,
):
    """
    Args:
        num_qubits: the number of qubits
        local_correlators: list of paulis
        theta: The QAOA angles.
        swap_strategy: selected swap strategy
        random_cut: A random cut, i.e., a series of 1 and 0 with the same length
            as the number of qubits. If qubit `i` has a `1` then we flip its
            initial state from `+` to `-`.
    """

    num_qubits = cost_operator.num_qubits

    if theta is not None:
        gamma = theta[: len(theta) // 2]
        beta = theta[len(theta) // 2 :]
        qaoa_layers = len(theta) // 2
    else:
        qaoa_layers = qaoa_layers

    # First, create the ansatz of 1 layer of QAOA without mixer

    if initial_state is None:
        initial_state = QuantumCircuit(num_qubits)

    if mixer_operator is None:
        mixer_operator = QuantumCircuit(num_qubits)

    qaoa_ansatz = QAOAAnsatz(cost_operator,
                             reps=1,
                             initial_state=initial_state,
                             mixer_operator=mixer_operator).decompose()

    # This will allow us to recover the permutation of the measurements that the swap introduce.
    qaoa_ansatz.measure_all()

    # Now, apply the swap strategy for commuting pauli evolution gates
    qaoa_ansatz = apply_swap_strategy(qaoa_ansatz, swap_strategy, edge_coloring)

    # Compute the measurement map (qubit to classical bit).
    # we will apply this for qaoa_layers % 2 == 1.

    if qaoa_layers % 2 == 1:
        meas_map = make_meas_map(qaoa_ansatz)
    else:
        meas_map = {idx: idx for idx in range(num_qubits)}

    qaoa_ansatz.remove_final_measurements()

    if theta is not None:
        # Finally, introduce the mixer circuit and add measurements following measurement map
        circuit = apply_qaoa_layers(qaoa_ansatz, meas_map, qaoa_layers, gamma, beta)
    else:
        # Finally, introduce the mixer circuit and add measurements following measurement map
        circuit = apply_qaoa_layers(qaoa_ansatz, meas_map, qaoa_layers)

    return circuit
