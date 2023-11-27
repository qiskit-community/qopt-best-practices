import json
import os
from unittest import TestCase

from qiskit.providers.fake_provider import FakeWashington

from qopt_best_practices.utils import build_max_cut_graph
from qopt_best_practices.qubit_selection import BackendEvaluator, find_lines


class TestQubitSelection(TestCase):
    """Unit test for QAOA workflow."""

    def setUp(self):
        super().setUp()

        # load data
        graph_file = os.path.join(os.path.dirname(__file__), "data/graph_2layers_0seed.json")

        with open(graph_file, "r") as f:
            data = json.load(f)

        self.mapped_paulis = [tuple(pauli) for pauli in data["paulis"]]
        self.mapped_graph = build_max_cut_graph(self.mapped_paulis)
        self.backend = FakeWashington()

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
