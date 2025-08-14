"""Tests for the cost layer."""

from unittest import TestCase

from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from qopt_best_practices.transpilation.cost_layer import get_cost_layer


class TestCostLayer(TestCase):
    """Test the construction of the cost layer."""

    def test_simple_cost(self):
        """Test the simple cost layer."""

        cost_op = SparsePauliOp.from_list([("ZZ", 1), ("IZ", 2)])

        cost_layer = get_cost_layer(cost_op)

        self.assertTrue(len(cost_layer.parameters), 1)
        self.assertTrue(cost_layer.parameters[0].name, "γ[0]")

    def test_parameterized_cost(self):
        """Test multi-objective style cost operators."""

        op1 = SparsePauliOp.from_list([("ZZ", 1), ("IZ", 2)])
        op2 = SparsePauliOp.from_list([("ZZ", 1), ("ZI", 2)])
        param1 = Parameter("c1")
        param2 = Parameter("c2")

        cost_op = param1 * op1 + param2 * op2

        cost_layer = get_cost_layer(cost_op)

        self.assertTrue(len( cost_layer.parameters), 3)
        for name in ["c1", "c2", "γ[0]"]:
            self.assertTrue(name in [param.name for param in cost_layer.parameters])
