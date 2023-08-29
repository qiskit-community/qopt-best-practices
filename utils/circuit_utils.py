from qiskit import QuantumCircuit, transpile
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
        N,# num. nodes
        local_correlators, # 3 from SAT mapping
        theta,
        random_cut=None
    ):
        """
        Args:
            theta: The QAOA angles.
            superposition: If True we initialized the qubits in the `+` state.
            random_cut: A random cut, i.e., a series of 1 and 0 with the same length
                as the number of qubits. If qubit `i` has a `1` then we flip its
                initial state from `+` to `-`.
            transpile_circ: If True, we transpile the circuit to the backend.
            remove_rz: If True then the rz gates in the cost Hamiltonian part
                of the circuit will be replaced with barriers. This makes it
                possible to efficiently simulate the circuit.
            apply_swaps: If True, the default, then we apply the swap pass manager.
                This can be set to false for noiseless simulators only.
        """
        gamma = theta[: len(theta) // 2]
        beta = theta[len(theta) // 2:]
        p = len(theta) // 2

        # First, create the Hamiltonian of 1 layer of QAOA
        hc_evo = QuantumCircuit(N)
        op = qi.SparsePauliOp.from_list(local_correlators)
        gamma_param = Parameter("g")
        hc_evo.append(PauliEvolutionGate(op, -gamma_param), range(N))

        # This will allow us to recover the permutation of the measurements that the swap introduce.
        hc_evo.measure_all()

        edge_coloring = {(idx, idx + 1): idx % 2 for idx in range(N)}

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(
                    SwapStrategy.from_line([i for i in range(N)]),
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

        # Compute the measurement map (qubit to classical bit). we will apply this for p % 2 == 1.
        if p % 2 == 1:
            meas_map = make_meas_map(hc_evo)
        else:
            meas_map = {idx: idx for idx in range(N)}

        hc_evo.remove_final_measurements()

        qc = QuantumCircuit(N)

        if random_cut is not None:
            for idx, coin_flip in enumerate(random_cut):
                if coin_flip == 1:
                    qc.x(idx)

        for i in range(p):
            bind_dict = {gamma_param: gamma[i]}
            bound_hc = hc_evo.assign_parameters(bind_dict)
            if i % 2 == 0:
                qc.append(bound_hc, range(N))
            else:
                qc.append(bound_hc.reverse_ops(), range(N))

            qc.rx(-2 * beta[i], range(N))

        creg = ClassicalRegister(N)
        qc.add_register(creg)

        for qidx, cidx in meas_map.items():
            qc.measure(qidx, cidx)

        return qc
