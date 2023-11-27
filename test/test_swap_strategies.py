"""Tests for Swap Strategies"""

from unittest import TestCase
import json
import os

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit.quantum_info import SparsePauliOp

from qopt_best_practices.swap_strategies import create_qaoa_swap_circuit


class TestSwapStrategies(TestCase):
    """Unit test for SWAP strategies functionality."""

    def setUp(self):
        super().setUp()

        # load data
        graph_file = os.path.join(os.path.dirname(__file__), "data/graph_2layers_0seed.json")

        with open(graph_file, "r") as file:
            data = json.load(file)

        self.mapped_paulis = [tuple(pauli) for pauli in data["paulis"]]

        self.hamiltonian = SparsePauliOp.from_list(self.mapped_paulis)

        self.swap_strategy = SwapStrategy.from_line(list(range(self.hamiltonian.num_qubits)))

    def test_qaoa_circuit(self):
        """Test building the QAOA circuit."""

        edge_coloring = {
            (idx, idx + 1): (idx + 1) % 2 for idx in range(self.hamiltonian.num_qubits)
        }

        for layers in range(1, 6):
            qaoa_circ = create_qaoa_swap_circuit(
                self.hamiltonian, self.swap_strategy, edge_coloring, qaoa_layers=layers
            )

            self.assertEqual(len(qaoa_circ.parameters), layers * 2)

    # TODO: expand tests
