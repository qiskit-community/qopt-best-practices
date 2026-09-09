"""Shared API for initial qubit mapping methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy


def if_num_to_real(
    value: complex | float | ParameterExpression,
) -> float | ParameterExpression:
    """Extract real part from numeric values, preserve ParameterExpression.
    Raises:
        TypeError: If value is not numeric or ParameterExpression.
    """
    if isinstance(value, ParameterExpression):
        return value
    if isinstance(value, (int, float, complex, np.number)):
        return float(np.real(value))
    raise TypeError(f"Expected numeric value or ParameterExpression, got {type(value).__name__}")


@dataclass
class InitialMappingResult:
    """Result returned by initial mapping implementations."""

    mapping: dict
    objective_value: int | float
    objective_name: str
    elapsed_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> int | float | None:
        """Best annealing cost when this result comes from simulated annealing."""
        if self.objective_name == "cost":
            return self.objective_value

        return self.metadata.get("cost")

    @property
    def num_swap_layers(self) -> int | None:
        """Minimum number of swap layers when this result comes from SAT mapping."""
        if self.objective_name == "num_swap_layers":
            return int(self.objective_value)

        return self.metadata.get("num_swap_layers")


class InitialMapping(ABC):
    """Abstract base class for initial qubit mapping methods."""

    @abstractmethod
    def find_initial_mapping(
        self,
        program_graph: nx.Graph,
        swap_strategy: SwapStrategy,
    ) -> InitialMappingResult:
        """Find an initial logical-to-physical qubit mapping."""

    def remap_graph(
        self,
        graph: nx.Graph | SparsePauliOp,
        swap_strategy: SwapStrategy,
    ) -> tuple[nx.Graph | SparsePauliOp, dict, InitialMappingResult] | tuple[None, None, None]:
        """Apply an initial mapping to a graph or cost operator.

        The returned 3-tuple is ``(remapped_graph, edge_map, result)``.
        The ``result`` entry is an :class:`InitialMappingResult` with a common
        shape across mapper implementations.
        """
        op_input = isinstance(graph, SparsePauliOp)

        if op_input:
            graph = self.op2graph(graph)

        try:
            result = self.find_initial_mapping(graph, swap_strategy)
        except ValueError:
            return None, None, None

        if result is None:
            return None, None, None

        edge_map = self.edge_map_from_result(result, graph.number_of_nodes())
        if edge_map is None:
            return None, None, None

        remapped_graph = nx.relabel_nodes(graph, edge_map)

        if op_input:
            return self.graph2op(remapped_graph), edge_map, result

        return remapped_graph, edge_map, result

    @staticmethod
    def edge_map_from_result(result: Any, n_logical: int) -> dict | None:
        """Extract a logical-to-physical edge map from a mapper result."""
        mapping = getattr(result, "mapping", None)

        if mapping is None:
            return None

        if isinstance(mapping, dict):
            return {k: v for k, v in mapping.items() if k < n_logical}

        return dict(mapping)

    @staticmethod
    def graph2op(graph: nx.Graph) -> SparsePauliOp:
        """Convert a graph into a sparse Pauli operator."""
        pauli_list = []
        for node1, node2, data in graph.edges(data=True):
            paulis = ["I"] * len(graph)
            paulis[node1], paulis[node2] = "Z", "Z"
            weight = data["weight"] if "weight" in data else 1.0
            pauli_list.append(("".join(paulis)[::-1], weight))

        return SparsePauliOp.from_list(pauli_list)

    @staticmethod
    def op2graph(operator: SparsePauliOp) -> nx.Graph:
        """Convert a cost operator to a graph."""
        graph, edges = nx.Graph(), []
        for pauli_str, weight in operator.to_list():
            edge = [idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"]

            if len(edge) == 1:
                edges.append((edge[0], edge[0], if_num_to_real(weight)))
            elif len(edge) == 2:
                edges.append((edge[0], edge[1], if_num_to_real(weight)))
            else:
                raise ValueError(f"The operator {operator} is not Quadratic.")

        graph.add_weighted_edges_from(edges)
        return graph
