"""Circuit utils"""

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, ClassicalRegister

from qiskit import quantum_info as qi
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)


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


def create_qaoa_circ_pauli_evolution(
    num_qubits: int,
    local_correlators: list[tuple[str, float]],
    theta: list[float],
    swap_strategy: SwapStrategy,
    random_cut: list[float] = None,
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
    gamma = theta[: len(theta) // 2]
    beta = theta[len(theta) // 2 :]
    qaoa_layers = len(theta) // 2

    # First, create the Hamiltonian of 1 layer of QAOA
    hc_evo = QuantumCircuit(num_qubits)
    pauli_op = qi.SparsePauliOp.from_list(local_correlators)
    gamma_param = Parameter("g")
    hc_evo.append(PauliEvolutionGate(pauli_op, -gamma_param), range(num_qubits))

    # This will allow us to recover the permutation of the measurements that the swap introduce.
    hc_evo.measure_all()

    edge_coloring = {(idx, idx + 1): idx % 2 for idx in range(num_qubits)}

    pm_pre = PassManager(
        [
            FindCommutingPauliEvolutions(),
            Commuting2qGateRouter(
                swap_strategy,
                edge_coloring,
            ),
        ]
    )

    # apply swaps
    hc_evo = pm_pre.run(hc_evo)

    basis_gates = ["rz", "sx", "x", "cx"]

    # Now transpile to sx, rz, x, cx basis
    hc_evo = transpile(hc_evo, basis_gates=basis_gates)

    # Replace Rz with zero rotations in cost Hamiltonian if desired
    # Deleted for now

    # Compute the measurement map (qubit to classical bit).
    # we will apply this for qaoa_layers % 2 == 1.
    if qaoa_layers % 2 == 1:
        meas_map = make_meas_map(hc_evo)
    else:
        meas_map = {idx: idx for idx in range(num_qubits)}

    hc_evo.remove_final_measurements()

    circuit = QuantumCircuit(num_qubits)

    if random_cut is not None:
        for idx, coin_flip in enumerate(random_cut):
            if coin_flip == 1:
                circuit.x(idx)

    for layer in range(qaoa_layers):
        bind_dict = {gamma_param: gamma[layer]}
        bound_hc = hc_evo.assign_parameters(bind_dict)
        if layer % 2 == 0:
            circuit.append(bound_hc, range(num_qubits))
        else:
            circuit.append(bound_hc.reverse_ops(), range(num_qubits))

        circuit.rx(-2 * beta[layer], range(num_qubits))

    creg = ClassicalRegister(num_qubits)
    circuit.add_register(creg)

    for qidx, cidx in meas_map.items():
        circuit.measure(qidx, cidx)

    return circuit
