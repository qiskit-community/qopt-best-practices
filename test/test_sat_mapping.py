"""Tests for SAT Mapping Utils"""

from unittest import TestCase
import json
import os
import networkx as nx

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.utils import build_max_cut_graph, build_max_cut_paulis
from qopt_best_practices.sat_mapping import SATMapper


class TestSwapStrategies(TestCase):
    """Unit test for SWAP strategies functionality."""

    def setUp(self):
        super().setUp()

        # load data
        graph_file = os.path.join(os.path.dirname(__file__), "data/graph_2layers_0seed.json")

        with open(graph_file, "r") as file:
            data = json.load(file)

        self.original_graph = nx.from_edgelist(data["Original graph"])
        self.original_paulis = build_max_cut_paulis(self.original_graph)

        self.mapped_paulis = [tuple(pauli) for pauli in data["paulis"]]
        self.mapped_graph = build_max_cut_graph(self.mapped_paulis)

        self.sat_mapping = {int(key): value for key, value in data["SAT mapping"].items()}
        self.min_k = data["min swap layers"]
        self.swap_strategy = SwapStrategy.from_line(list(range(len(self.original_graph.nodes))))
        self.basic_graphs = [nx.path_graph(5), nx.cycle_graph(7)]

    def test_find_initial_mappings(self):
        """Test find_initial_mappings"""

        mapper = SATMapper()

        results = mapper.find_initial_mappings(self.original_graph, self.swap_strategy)
        min_k = min((k for k, v in results.items() if v.satisfiable))
        # edge_map = dict(results[min_k].mapping)

        # edge maps are not equal, but same min_k
        self.assertEqual(min_k, self.min_k)

        # Find better test
        # self.assertEqual(edge_map, self.sat_mapping)

    def test_remap_graph_with_sat(self):
        """Test remap_graph_with_sat"""

        mapper = SATMapper()

        remapped_g, _, _ = mapper.remap_graph_with_sat(
            graph=self.original_graph, swap_strategy=self.swap_strategy
        )

        self.assertTrue(nx.is_isomorphic(remapped_g, self.mapped_graph))

    def test_deficient_strategy(self):
        """Test that the SAT mapper works when the SWAP strategy is deficient.

        Note: a deficient strategy does not result in full connectivity but
        may still be useful.
        """
        cmap = CouplingMap([(idx, idx + 1) for idx in range(10)])

        # This swap strategy is deficient but can route the graph below.
        swaps = (
            ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
            ((1, 2), (3, 4), (5, 6), (7, 8)),
            ((2, 3), (4, 5), (6, 7), (8, 9)),
            (),
            (),
            (),
            (),
            (),
        )
        swap_strategy = SwapStrategy(cmap, swaps)
        graph = nx.random_regular_graph(3, 10, seed=2)

        mapper = SATMapper()

        _, permutation, min_layer = mapper.remap_graph_with_sat(graph, swap_strategy)

        # Spot check a few permutations.
        self.assertEqual(permutation[0], 9)
        self.assertEqual(permutation[8], 1)

        # Crucially, if the `connectivity_matrix` in `find_initial_mappings` we get a wrong result.
        self.assertEqual(min_layer, 3)

    def test_full_connectivity(self):
        """Test that the SAT mapper works when the SWAP strategy has full connectivity."""
        graph = nx.random_regular_graph(3, 6, seed=1)
        swap_strategy = SwapStrategy.from_line(list(range(6)))
        sat_mapper = SATMapper()
        _, _, min_sat_layers = sat_mapper.remap_graph_with_sat(
            graph=graph,
            swap_strategy=swap_strategy,
        )
        self.assertEqual(min_sat_layers, 4)

    def test_unable_to_remap(self):
        """Test that the SAT mapper works when the SWAP strategy is unable to remap."""
        graph = nx.random_regular_graph(3, 6, seed=1)
        swap_strategy = SwapStrategy.from_line(list(range(6)))
        cmap = CouplingMap([(idx, idx + 1) for idx in range(5)])
        swap_strategy = SwapStrategy(cmap, [])
        sat_mapper = SATMapper()
        remapped_g, edge_map, min_sat_layers = sat_mapper.remap_graph_with_sat(
            graph=graph,
            swap_strategy=swap_strategy,
        )
        self.assertIsNone(remapped_g)
        self.assertIsNone(edge_map)
        self.assertIsNone(min_sat_layers)
