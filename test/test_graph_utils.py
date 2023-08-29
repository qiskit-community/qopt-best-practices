from unittest import TestCase
import networkx as nx

from qopt_best_practices.utils import build_graph, build_paulis


class TestGraphRoundTrip(TestCase):
    """Test that we can convert between graph and Paulis."""

    @staticmethod
    def _test_edge_equality(g: nx.Graph, h: nx.Graph):
        """Test equality of edges."""
        if len(g.edges) != len(h.edges):
            return False

        g_set = set(g.edges)
        for edge in h.edges:
            if edge not in g_set and edge[::-1] not in g_set:
                return False

        return True

    def test_round_trip(self):
        """Test that we can easily round-trip Pauli the graphs."""

        for seed in range(5):
            graph1 = nx.random_regular_graph(3, 10, seed=seed)
            graph2 = build_graph(build_paulis(graph1))

            self.assertTrue(self._test_edge_equality(graph1, graph2))
