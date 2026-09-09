"""Tests for the simulated annealing mapper."""

from unittest import TestCase

import networkx as nx

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.qubit_mapping import (
    InitialMapping,
    InitialMappingResult,
    SAMapper,
    SimulatedAnnealingMapper,
)


class TestSimulatedAnnealingMapping(TestCase):
    """Unit tests for the simulated annealing mapping."""

    def setUp(self):
        super().setUp()

        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0)])
        self.swap_strategy = SwapStrategy.from_line(list(range(self.graph.number_of_nodes())))

    @staticmethod
    def _build_mapper():
        return SimulatedAnnealingMapper(max_iter=3000, max_restarts=2, seed=0)

    @staticmethod
    def _weighted_edges(graph):
        return sorted((min(u, v), max(u, v), weight) for u, v, weight in graph.edges(data="weight"))

    def test_graph_to_operator_round_trip(self):
        """Test graph2op/op2graph preserve weighted edges."""

        mapper = SimulatedAnnealingMapper()

        operator = mapper.graph2op(self.graph)
        round_trip_graph = mapper.op2graph(operator)

        self.assertEqual(
            self._weighted_edges(round_trip_graph),
            self._weighted_edges(self.graph),
        )

    def test_find_initial_mapping_returns_valid_mapping(self):
        """Test find_initial_mapping returns a logical-to-physical permutation."""

        mapper = self._build_mapper()

        result = mapper.find_initial_mapping(self.graph, self.swap_strategy)

        self.assertEqual(set(result.mapping), set(self.graph.nodes))
        self.assertEqual(set(result.mapping.values()), set(range(self.graph.number_of_nodes())))
        self.assertGreaterEqual(result.cost, 0)
        self.assertEqual(result.num_swap_layers, result.cost)
        self.assertGreaterEqual(result.elapsed_time, 0.0)

    def test_path_graph_needs_no_swap_layers(self):
        """Test the annealer finds a zero-swap-layer mapping for a path graph."""

        mapper = self._build_mapper()

        result = mapper.find_initial_mapping(self.graph, self.swap_strategy)

        self.assertEqual(result.cost, 0)

    def test_seed_reproducibility(self):
        """Test that the same seed gives the same mapping."""

        result_a = self._build_mapper().find_initial_mapping(self.graph, self.swap_strategy)
        result_b = self._build_mapper().find_initial_mapping(self.graph, self.swap_strategy)

        self.assertEqual(result_a.mapping, result_b.mapping)
        self.assertEqual(result_a.cost, result_b.cost)

    def test_mapping_quality_on_regular_graph(self):
        """Test the achieved number of swap layers on a 3-regular graph.

        The maximum distance-matrix entry over the mapped program edges must
        equal the reported cost, and lie well below the worst case.
        """

        graph = nx.random_regular_graph(3, 14, seed=2)
        swap_strategy = SwapStrategy.from_line(list(range(14)))
        mapper = SimulatedAnnealingMapper(seed=1)

        result = mapper.find_initial_mapping(graph, swap_strategy)

        dist = swap_strategy.distance_matrix
        achieved = max(int(dist[result.mapping[u]][result.mapping[v]]) for u, v in graph.edges())
        self.assertEqual(achieved, result.cost)
        self.assertLessEqual(result.cost, 7)

    def test_arbitrary_node_labels(self):
        """Test that non-integer node labels are supported."""

        graph = nx.Graph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        swap_strategy = SwapStrategy.from_line(list(range(4)))
        mapper = self._build_mapper()

        result = mapper.find_initial_mapping(graph, swap_strategy)

        self.assertEqual(set(result.mapping), {"a", "b", "c", "d"})
        self.assertEqual(set(result.mapping.values()), set(range(4)))

    def test_more_physical_than_logical_qubits(self):
        """Test mapping onto a swap strategy with spare physical qubits."""

        swap_strategy = SwapStrategy.from_line(list(range(6)))
        mapper = self._build_mapper()

        result = mapper.find_initial_mapping(self.graph, self.swap_strategy)
        result_padded = mapper.find_initial_mapping(self.graph, swap_strategy)

        self.assertEqual(set(result_padded.mapping), set(self.graph.nodes))
        self.assertEqual(len(set(result_padded.mapping.values())), len(self.graph.nodes))
        self.assertLessEqual(result_padded.cost, result.cost)

    def test_shared_initial_mapping_api(self):
        """Test SAMapper implements the shared initial mapping API."""

        mapper = SAMapper(max_iter=3000, max_restarts=2, seed=0)

        remapped_graph, edge_map, remap_result = mapper.remap_graph(self.graph, self.swap_strategy)

        self.assertIsInstance(mapper, InitialMapping)
        self.assertIsInstance(SimulatedAnnealingMapper(), SAMapper)
        self.assertIsInstance(remap_result, InitialMappingResult)
        self.assertEqual(remap_result.objective_name, "cost")
        self.assertEqual(set(edge_map), set(self.graph.nodes))
        self.assertEqual(set(edge_map.values()), set(range(self.graph.number_of_nodes())))
        self.assertGreaterEqual(remap_result.cost, 0)
        self.assertEqual(
            self._weighted_edges(remapped_graph),
            self._weighted_edges(nx.relabel_nodes(self.graph, edge_map)),
        )

    def test_remap_graph_with_sa_matches_find_initial_mapping(self):
        """Test remap_graph_with_sa uses the same mapping returned by find_initial_mapping."""

        mapper = self._build_mapper()

        result = mapper.find_initial_mapping(self.graph, self.swap_strategy)

        remapped_graph, edge_map, remap_result = mapper.remap_graph_with_sa(
            self.graph, self.swap_strategy
        )

        self.assertEqual(edge_map, result.mapping)
        self.assertEqual(remap_result.cost, result.cost)
        self.assertEqual(
            self._weighted_edges(remapped_graph),
            self._weighted_edges(nx.relabel_nodes(self.graph, edge_map)),
        )

    def test_remap_graph_with_sa_accepts_sparse_pauli_op(self):
        """Test remap_graph_with_sa accepts cost operators as input."""

        mapper = self._build_mapper()
        operator = mapper.graph2op(self.graph)

        remapped_op, edge_map, result = mapper.remap_graph_with_sa(operator, self.swap_strategy)

        self.assertIsInstance(remapped_op, SparsePauliOp)
        self.assertIsInstance(result, InitialMappingResult)
        self.assertEqual(set(edge_map), set(self.graph.nodes))
        self.assertEqual(set(edge_map.values()), set(range(self.graph.number_of_nodes())))
        self.assertGreaterEqual(result.cost, 0)
        self.assertEqual(
            self._weighted_edges(mapper.op2graph(remapped_op)),
            self._weighted_edges(nx.relabel_nodes(self.graph, edge_map)),
        )

    def test_find_initial_mapping_raises_with_too_few_physical_qubits(self):
        """Test find_initial_mapping rejects oversized program graphs."""

        graph = nx.path_graph(6)
        cmap = CouplingMap([(idx, idx + 1) for idx in range(4)])
        swap_strategy = SwapStrategy(cmap, [])
        mapper = self._build_mapper()

        with self.assertRaises(ValueError):
            mapper.find_initial_mapping(graph, swap_strategy)

    def test_remap_graph_with_sa_returns_none_when_mapping_fails(self):
        """Test remap_graph_with_sa returns a None triple on failure."""

        graph = nx.path_graph(6)
        cmap = CouplingMap([(idx, idx + 1) for idx in range(4)])
        swap_strategy = SwapStrategy(cmap, [])
        mapper = self._build_mapper()

        remapped_graph, edge_map, result = mapper.remap_graph_with_sa(graph, swap_strategy)

        self.assertIsNone(remapped_graph)
        self.assertIsNone(edge_map)
        self.assertIsNone(result)
