from unittest import TestCase
import json

import networkx as nx

from qiskit.providers.fake_provider import FakeWashington
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.utils import build_graph, build_paulis
from qopt_best_practices.utils import create_qaoa_circ_pauli_evolution

from qopt_best_practices.sat_mapping import SATMapper
from qopt_best_practices.qubit_selection import BackendEvaluator


class TestFullWorkflowLine(TestCase):

    """Unit test for QAOA workflow."""

    def setUp(self):
        super().setUp()

        # load data
        graph_file = "data/graph_2layers_0seed.json"

        with open(graph_file, "r") as f:
            data = json.load(f)

        self.original_graph = nx.from_edgelist(data["Original graph"])
        self.original_paulis = build_paulis(self.original_graph)

        self.mapped_paulis = [tuple(pauli) for pauli in data["paulis"]]
        self.mapped_graph = build_graph(self.mapped_paulis)

        self.sat_mapping = {
            int(key): value for key, value in data["SAT mapping"].items()
        }
        self.min_swap_layers = data["min swap layers"]

        self.backend = FakeWashington()
        self.swap_strategy = SwapStrategy.from_line(
            [i for i in range(len(self.original_graph.nodes))]
        )

    def test_sat_mapping(self):
        """Test SAT mapping"""

        sat_mapper = SATMapper()
        min_k, edge_map, paulis = sat_mapper.remap_graph_with_sat(
            graph=self.original_graph, swap_strategy=self.swap_strategy
        )

        self.assertEqual(min_k, self.min_swap_layers)
        self.assertDictEqual(edge_map, self.sat_mapping)
        self.assertEqual(set(paulis), set(self.mapped_paulis))

    def test_qubit_selection(self):
        """Test backend evaluation"""

        path_finder = BackendEvaluator(self.backend)

        path, _ = path_finder.evaluate(len(self.mapped_graph))

        # this is just a placeholder, get proper data to test
        expected_path = [30, 31, 32, 36, 51, 50, 49, 48, 47, 35]

        self.assertEqual(set(path), set(expected_path))

    def test_circuit_construction(self):
        """Test circuit construction"""

        theta = [1, 1, 0, 1]
        qaoa_circ = create_qaoa_circ_pauli_evolution(
            len(self.mapped_graph), self.mapped_paulis, theta, self.swap_strategy
        )

        print(qaoa_circ)
