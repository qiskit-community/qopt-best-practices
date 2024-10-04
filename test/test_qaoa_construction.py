"""Test the construction of the QAOA ansatz."""


from unittest import TestCase

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation.preset_qaoa_passmanager import qaoa_swap_strategy


class TestQAOAConstruction(TestCase):
    """Test the construction of the QAOA ansatz."""

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

        self.cost_op_circ = transpile(cost_op, basis_gates=["sx", "cx", "x", "rz"])

        self.cost_op = SparsePauliOp.from_list([("IIZZ", 1), ("ZZII", 1), ("ZIIZ", 1)])

        self.config = {
            "swap_strategy": SwapStrategy.from_line(list(range(4))),
            "edge_coloring": {(idx, idx + 1): (idx + 1) % 2 for idx in range(4)},
        }

    def test_depth_one(self):
        """Compare the pass with the SWAPs and ensure the measurements are ordered properly."""
        qaoa_pm = qaoa_swap_strategy_pm(self.config)

        cost_op_circ = QAOAAnsatz(
            self.cost_op, initial_state=QuantumCircuit(4), mixer_operator=QuantumCircuit(4)
        ).decompose(reps=1)

        ansatz = qaoa_pm.run(cost_op_circ)

        # 1. Check the measurement map
        qreg = ansatz.qregs[0]
        creg = ansatz.cregs[0]

        expected_meas_map = {0: 1, 1: 0, 2: 3, 3: 2}

        for inst in ansatz.data:
            if inst.operation.name == "measure":
                qubit = qreg.index(inst.qubits[0])
                cbit = creg.index(inst.clbits[0])
                self.assertEqual(cbit, expected_meas_map[qubit])

        # 2. Check the expectation value. Note that to use the estimator we need to
        # Remove the final measurements and correspondingly permute the cost op.
        ansatz.remove_final_measurements(inplace=True)
        permuted_cost_op = SparsePauliOp.from_list([("IIZZ", 1), ("ZZII", 1), ("IZZI", 1)])
        value = self.estimator.run([(ansatz, permuted_cost_op, [1, 2])]).result()[0].data.evs

        library_ansatz = QAOAAnsatz(self.cost_op, reps=1)
        library_ansatz = transpile(library_ansatz, basis_gates=["cx", "rz", "rx", "h"])

        expected = self.estimator.run([(library_ansatz, self.cost_op, [1, 2])]).result()[0].data.evs

        self.assertAlmostEqual(value, expected)

    def test_depth_two_qaoa_pass(self):
        """Compare the pass with the SWAPs to an all-to-all construction.

        Note: this test only works as is because p is even and we don't have the previous
        passes to give us the qubit permutations.
        """
        qaoa_pm = PassManager([QAOAConstructionPass(num_layers=2)])

        ansatz = qaoa_pm.run(self.cost_op_circ)
        ansatz.remove_final_measurements(inplace=True)

        value = self.estimator.run([(ansatz, self.cost_op, [1, 2, 3, 4])]).result()[0].data.evs

        library_ansatz = QAOAAnsatz(self.cost_op, reps=2)
        library_ansatz = transpile(library_ansatz, basis_gates=["cx", "rz", "rx", "h"])

        expected = (
            self.estimator.run([(library_ansatz, self.cost_op, [1, 2, 3, 4])]).result()[0].data.evs
        )

        self.assertAlmostEqual(value, expected)
