"""Circuit utils"""

from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp

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

    This allows us to account for the swapping order introduced by the SwapStrategy.
    """
    creg = circuit.cregs[0]
    qreg = circuit.qregs[0]

    meas_map = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            meas_map[qreg.index(inst.qubits[0])] = creg.index(inst.clbits[0])

    return meas_map


def apply_swap_strategy(
    circuit: QuantumCircuit,
    swap_strategy: SwapStrategy,
    edge_coloring: dict[tuple[int, int], int] | None = None,
) -> QuantumCircuit:
    """Transpile with a SWAP strategy.

    Returns:
        A quantum circuit transpiled with the given swap strategy.
    """

    pm_pre = PassManager(
        [
            FindCommutingPauliEvolutions(),
            Commuting2qGateRouter(
                swap_strategy,
                edge_coloring,
            ),
        ]
    )
    return pm_pre.run(circuit)


def apply_qaoa_layers(
    cost_layer: QuantumCircuit,
    meas_map: dict,
    num_layers: int,
    gamma: list[float] | ParameterVector = None,
    beta: list[float] | ParameterVector = None,
    initial_state: QuantumCircuit = None,
    mixer: QuantumCircuit = None,
):
    """Construct the QAOA circuit.

    First, the initial state is applied. If `initial_state` is None we begin in the
    initial superposition state. Next, we alternate between layers of the cot operator
    and the mixer. The cost operator is alternatively applied in order and in reverse
    instruction order. This allows us to apply the swap-strategy on odd `p` layers
    and undo the swap strategy on even `p` layers.
    """

    num_qubits = cost_layer.num_qubits
    new_circuit = QuantumCircuit(num_qubits, num_qubits)

    if initial_state is not None:
        new_circuit.append(initial_state, range(num_qubits))
    else:
        # all h state by default
        new_circuit.h(range(num_qubits))

    if gamma is None or beta is None:
        gamma = ParameterVector("γ", num_layers)
        if mixer is None or mixer.num_parameters == 0:
            beta = ParameterVector("β", num_layers)
        else:
            beta = ParameterVector("β", num_layers * mixer.num_parameters)

    if mixer is not None:
        mixer_layer = mixer
    else:
        mixer_layer = QuantumCircuit(num_qubits)
        mixer_layer.rx(beta[0], range(num_qubits))

    for layer in range(num_layers):
        bind_dict = {cost_layer.parameters[0]: gamma[layer]}
        cost_layer_ = cost_layer.assign_parameters(bind_dict)
        bind_dict = {
            mixer_layer.parameters[i]: beta[layer + i]
            for i in range(mixer_layer.num_parameters)
        }
        layer_mixer = mixer_layer.assign_parameters(bind_dict)

        if layer % 2 == 0:
            new_circuit.append(cost_layer_, range(num_qubits))
        else:
            new_circuit.append(cost_layer_.reverse_ops(), range(num_qubits))

        new_circuit.append(layer_mixer, range(num_qubits))

    for qidx, cidx in meas_map.items():
        new_circuit.measure(qidx, cidx)

    return new_circuit


def create_qaoa_swap_circuit(
    cost_operator: list[tuple[str, float]],
    swap_strategy: SwapStrategy,
    edge_coloring: dict = None,
    theta: list[float] = None,
    qaoa_layers: int = 1,
    initial_state: QuantumCircuit = None,
    mixer: QuantumCircuit = None,
):
    """
    This method can only handle circuits with only 1-qubit or 2-qubit gates, due to the limitation in the function `apply_swap_strategy`, which can only handle 2-qubit gates. Given this constraint, we still have to treat the 1-qubit gates and 2-qubit gates separately. Suppose H = H1 + H2, where H1 has only 1-qubit gates, and H2 only 2-qubit gates.
    
    Strategy is
        - create correspponding circuits for both H1 and H2
        - `apply_swap_strategy` on the circuit of H2
        - combine the two circuits
        
    Args:
        num_qubits: the number of qubits
        local_correlators: list of paulis
        theta: The QAOA angles.
        swap_strategy: selected swap strategy
        random_cut: A random cut, i.e., a series of 1 and 0 with the same length
            as the number of qubits. If qubit `i` has a `1` then we flip its
            initial state from `+` to `-`.
    """

    # Save the parameters of the original, total Hamiltonian 
    num_qubits = cost_operator.num_qubits
    gate_list = cost_operator.paulis
    weights_list = cost_operator.coeffs
    
    # Prepare the separate two parts of the total Hamiltonian
    cost_operator_order1_only = QuantumCircuit(num_qubits)
    cost_operator_order2_only = []

    # Create H2 as an operator
    for pauli, gate_weight in zip(gate_list, weights_list):
        if sum(pauli.x) != 0 or sum(pauli.z) > 2:
            raise Exception("This method can only handle first order and second order Pauli Z terms.")
        if sum(pauli.z) == 2:
            cost_operator_order2_only.append((pauli, gate_weight))
    cost_operator_order2_only = SparsePauliOp(list(zip(*cost_operator_order2_only))[0], list(zip(*cost_operator_order2_only))[1])

    # Then, create H2 as a circuit, using the ansatz of 1 layer of QAOA without mixer
    cost_layer_order2_only = QAOAAnsatz(
        cost_operator_order2_only,
        reps=1,
        initial_state=QuantumCircuit(num_qubits),
        mixer_operator=QuantumCircuit(num_qubits),
    ).decompose()

    # Save the gamma parameters used in H2, which will be the same as for the ansatz of H1
    gamma = cost_layer_order2_only.parameters[0]

    # Create ansatz for H1 
    for pauli, gate_weight in zip(gate_list, weights_list):
        
        if sum(pauli.z) == 1:
            print("pauli, gate_weight, gamma=", (pauli, gate_weight, gamma))
            qubit_index = np.where(pauli.z == True)[0][0] 
            cost_operator_order1_only.rz(gate_weight*gamma, qubit_index)
    
    # Before applying the swap_strategy to H2, we want to recover the permutation of the measurements that the swap introduce.
    cost_layer_order2_only.measure_all()
    
    # Now, apply the swap strategy for commuting pauli evolution gates
    cost_layer_order2_only  = apply_swap_strategy(cost_layer_order2_only, swap_strategy, edge_coloring)

    # Combine the H1 ansatz with the optimized H2 ansatz
    cost_layer = QuantumCircuit(num_qubits)
    cost_layer.compose(cost_operator_order1_only, inplace=True)
    cost_layer.compose(cost_layer_order2_only, inplace=True)

    # Compute the measurement map (qubit to classical bit).
    # we will apply this for qaoa_layers % 2 == 1.
    if qaoa_layers % 2 == 1:
        meas_map = make_meas_map(cost_layer)
    else:
        meas_map = {idx: idx for idx in range(num_qubits)}
    
    cost_layer.remove_final_measurements()

    # Assign variational parameters
    if theta is not None:
        gamma = theta[: len(theta) // 2]
        beta = theta[len(theta) // 2 :]
        qaoa_layers = len(theta) // 2
    else:
        gamma = beta = None
        qaoa_layers = qaoa_layers

    # Finally, introduce the mixer circuit and add measurements following measurement map
    circuit = apply_qaoa_layers(
        cost_layer, meas_map, qaoa_layers, gamma, beta, initial_state, mixer
    )

    return circuit
