"""Test the preparation of the cost layer for the swap netwrok routing."""

from unittest import TestCase

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager

from qopt_best_practices.transpilation.cost_layer import get_cost_layer
from qopt_best_practices.transpilation.prepare_cost_layer import PrepareCostLayer


class TestPrepareCostLayer(TestCase):
    """Test the prepare cost layer pass."""

    def test_simple(self):
        """Test with single and two-qubit terms."""

        cost_op = SparsePauliOp.from_list([("IZZ", 1), ("ZIZ", 1), ("IIZ", 1)])

        pm_ = PassManager([PrepareCostLayer()])

        circ = pm_.run(get_cost_layer(cost_op))

        self.assertEqual(len(circ.data), 2)

        inst_name = [inst.name for inst in circ.data]

        self.assertTrue("rz" in inst_name)
        self.assertTrue("commuting_2q_block" in inst_name)
