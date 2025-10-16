"""Test the preset QAOA pass manager."""

from unittest import TestCase

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
from qopt_best_practices.transpilation.annotated_transpilation_passes import UnrollBoxes
from qopt_best_practices.transpilation.generate_preset_qaoa_pass_manager import (
    generate_preset_qaoa_pass_manager,
)


class TestPresetQAOAPassManager(TestCase):
    """Tests for the preset QAOA pass manager."""

    def setUp(self):
        """Set up re-used variables."""
        self.estimator = StatevectorEstimator()
        gamma = Parameter("γ")
        cost_op = QuantumCircuit(4)
        cost_op.rzz(2 * gamma, 0, 1)
        cost_op.rzz(2 * gamma, 2, 3)
        cost_op.swap(0, 1)
        cost_op.swap(2, 3)
        cost_op.rzz(2 * gamma, 1, 2)

        self.cost_op = SparsePauliOp.from_list([("IIZZ", 1), ("ZZII", 1), ("ZIIZ", 1)])

        self._swap_strategy = SwapStrategy.from_line(list(range(4)))
        self._edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(4)}

        cmap = CouplingMap([(0, 1), (1, 2), (2, 3)])
        self._backend = GenericBackendV2(
            num_qubits=4, coupling_map=cmap, basis_gates=["x", "sx", "cz", "id", "rz"], seed=0
        )

    def test_depth_one(self):
        """Compare the pass with the SWAPs and ensure the measurements are ordered properly."""
        qaoa_pm = generate_preset_qaoa_pass_manager(
            self._backend,
            swap_strategy=self._swap_strategy,
            edge_coloring=self._edge_coloring,
        )

        ansatz = annotated_qaoa_ansatz(self.cost_op)
        ansatz.measure_all()

        isa_ansatz = qaoa_pm.run(ansatz)

        # 1. Check the measurement map
        qreg = isa_ansatz.qregs[0]
        creg = isa_ansatz.cregs[0]

        expected_meas_map = {0: 1, 1: 0, 2: 3, 3: 2}

        for inst in isa_ansatz.data:
            if inst.operation.name == "measure":
                qubit = qreg.index(inst.qubits[0])
                cbit = creg.index(inst.clbits[0])
                self.assertEqual(cbit, expected_meas_map[qubit])

        # 2. Check the expectation value. Note that to use the estimator we need to
        # Remove the final measurements and correspondingly permute the cost op.

        # a) Baseline: energy evaluation on an all-to-all backend
        pm_ = PassManager([UnrollBoxes()])
        sim_ansatz = pm_.run(ansatz)
        sim_ansatz.remove_final_measurements(inplace=True)
        expected = self.estimator.run([(sim_ansatz, self.cost_op, [1, 2])]).result()[0].data.evs

        # b) Energy evaluation of the routed ansatz with the preset pass manager
        permuted_cost_op = SparsePauliOp.from_list([("IIZZ", 1), ("ZZII", 1), ("IZZI", 1)])
        isa_ansatz.remove_final_measurements(inplace=True)
        value = self.estimator.run([(isa_ansatz, permuted_cost_op, [1, 2])]).result()[0].data.evs

        self.assertAlmostEqual(value, expected)
