from unittest import TestCase
import json

import networkx as nx

from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.utils import build_graph, build_paulis
from qopt_best_practices.swap_strategies import *


class TestSwapStrategies(TestCase):
    """Unit test for SWAP strategies functionality."""

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

        self.backend = FakeGuadalupe()
        self.swap_strategy = SwapStrategy.from_line(
            [i for i in range(len(self.original_graph.nodes))]
        )
