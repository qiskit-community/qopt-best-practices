"""Tests for Graph Utils"""

from unittest import TestCase
import networkx as nx

from qopt_best_practices.utils import build_max_cut_graph, build_max_cut_paulis


class TestGraphRoundTrip(TestCase):
    """Test that we can convert between graph and Paulis."""

    @staticmethod
    def _test_edge_equality(graph_1: nx.Graph, graph_2: nx.Graph):
        """Test equality of edges."""
        if len(graph_1.edges) != len(graph_2.edges):
            return False

        g_set = set(graph_1.edges)
        for edge in graph_2.edges:
            if edge not in g_set and edge[::-1] not in g_set:
                return False

        return True

    def test_round_trip(self):
        """Test that we can easily round-trip Pauli the graphs."""

        for seed in range(5):
            graph1 = nx.random_regular_graph(3, 10, seed=seed)
            graph2 = build_max_cut_graph(build_max_cut_paulis(graph1))

            self.assertTrue(self._test_edge_equality(graph1, graph2))

    def test_weighted_graph(self):
        """Test the construction of a weighted graph."""

        graph = build_max_cut_graph([("IIZZ", 1), ("IZZI", -1), ("ZIZI", 1)])

        self.assertEqual(graph.get_edge_data(0, 1)["weight"], 1)
        self.assertEqual(graph.get_edge_data(1, 2)["weight"], -1)
        self.assertEqual(graph.get_edge_data(1, 3)["weight"], 1)
