from unittest import TestCase
import json

import networkx as nx

from qiskit.providers.fake_provider import FakeWashington
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.utils import build_graph, build_paulis
from qopt_best_practices.swap_strategies import *

from qopt_best_practices.qubit_selection import (
    BackendEvaluator,
    find_lines,
    evaluate_fidelity,
)


class TestQubitSelection(TestCase):

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

    def test_find_lines(self):
        """Test backend evaluation"""

        paths = find_lines(len(self.mapped_graph), self.backend)

        self.assertEqual(len(paths), 1237)
        self.assertEqual(len(paths[0]), len(self.mapped_graph))
        self.assertIsInstance(paths[0][0], int)

    def test_qubit_selection(self):
        """Test backend evaluation"""

        path_finder = BackendEvaluator(self.backend)

        path, fidelity, num_subsets = path_finder.evaluate(len(self.mapped_graph))

        expected_path = [30, 31, 32, 36, 51, 50, 49, 48, 47, 35]

        self.assertEqual(set(path), set(expected_path))
