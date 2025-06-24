"""Tests for Qubit Selection Utils"""

import json
import os
from unittest import TestCase

from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeHanoiV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import __version__ as qiskit_version

from qopt_best_practices.utils import build_max_cut_graph
from qopt_best_practices.qubit_selection import BackendEvaluator, find_lines


class TestQubitSelection(TestCase):
    """Unit test for QAOA workflow."""

    def setUp(self):
        super().setUp()

        # load data
        graph_file = os.path.join(os.path.dirname(__file__), "data/graph_2layers_0seed.json")
        with open(graph_file, "r") as file:
            data = json.load(file)

        # 10 qubit graph
        self.mapped_paulis = [tuple(pauli) for pauli in data["paulis"]]
        self.mapped_graph = build_max_cut_graph(self.mapped_paulis)
        self.backend = FakeSherbrooke()
        self.major_version = int(qiskit_version.split(".")[0])

    def test_find_lines(self):
        """Test backend evaluation"""
        paths = find_lines(len(self.mapped_graph), self.backend)

        self.assertEqual(len(paths), 1336)
        self.assertEqual(len(paths[0]), len(self.mapped_graph))
        self.assertIsInstance(paths[0][0], int)

    def test_find_lines_directed(self):
        """Test backend with directed (asymmetric) coupling map"""
        directed_fake_backend = GenericBackendV2(
            4, coupling_map=[[0, 1], [1, 2], [3, 2], [3, 0]], seed=0
        )
        lines = find_lines(3, backend=directed_fake_backend)

        expected_lines = [[0, 1, 2], [0, 3, 2], [1, 2, 3], [1, 0, 3]]
        self.assertEqual(set(tuple(i) for i in lines), set(tuple(i) for i in expected_lines))

    def test_qubit_selection(self):
        """Test backend evaluation"""
        path_finder = BackendEvaluator(self.backend)
        path, _, _ = path_finder.evaluate(len(self.mapped_graph))

        if self.major_version < 2:
            expected_path = [33, 39, 40, 72, 41, 81, 53, 60, 61, 62]
        else:
            expected_path = [33, 39, 40, 41, 53, 60, 59, 58, 19, 20]

        self.assertEqual(set(path), set(expected_path))

    def test_qubit_selection_v2(self):
        """Test backend evaluation for 10 qubit line"""
        path_finder = BackendEvaluator(FakeHanoiV2())
        path, _, _ = path_finder.evaluate(len(self.mapped_graph))
        if self.major_version < 2:
            expected_path = [8, 9, 11, 12, 13, 14, 15, 18, 21, 23]
        else:
            expected_path = [1, 2, 3, 4, 5, 6, 10, 12, 13, 14]
        self.assertEqual(set(path), set(expected_path))
