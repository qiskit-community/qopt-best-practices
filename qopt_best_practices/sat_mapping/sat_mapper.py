"""A class to solve the SWAP gate insertion initial mapping problem
using the SAT approach from https://arxiv.org/abs/2212.05666.
"""

from __future__ import annotations
from typing import Union

from dataclasses import dataclass
from itertools import combinations
from threading import Timer

import networkx as nx
import numpy as np

from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy


@dataclass
class SATResult:
    """A data class to hold the result of a SAT solver."""

    satisfiable: bool  # Satisfiable is True if the SAT model could be solved in a given time.
    solution: dict  # The solution to the SAT problem if it is satisfiable.
    mapping: list  # The mapping of nodes in the pattern graph to nodes in the target graph.
    elapsed_time: float  # The time it took to solve the SAT model.


class SATMapper:
    r"""A class to introduce a SAT-approach to solve
    the initial mapping problem in SWAP gate insertion for commuting gates.

    When this pass is run on a DAG it will look for the first instance of
    :class:`.Commuting2qBlock` and use the program graph :math:`P` of this block of gates to
    find a layout for a given swap strategy. This layout is found with a
    binary search over the layers :math:`l` of the swap strategy. At each considered layer
    a subgraph isomorphism problem formulated as a SAT is solved by a SAT solver. Each instance
    is whether it is possible to embed the program graph :math:`P` into the effective
    connectivity graph :math:`C_l` that is achieved by applying :math:`l` layers of the
    swap strategy to the coupling map :math:`C_0` of the backend. Since solving SAT problems
    can be hard, a ``time_out`` fixes the maximum time allotted to the SAT solver for each
    instance. If this time is exceeded the considered problem is deemed unsatisfiable and
    the binary search proceeds to the next number of swap layers :math:``l``.
    """

    def __init__(self, timeout: int = 60):
        """Initialize the SATMapping.

        Args:
            timeout: The allowed time in seconds for each iteration of the SAT solver. This
                variable defaults to 60 seconds.
        """
        self.timeout = timeout

    def find_initial_mappings(
        self,
        program_graph: nx.Graph,
        swap_strategy: SwapStrategy,
        min_layers: int | None = None,
        max_layers: int | None = None,
    ) -> dict[int, SATResult]:
        r"""Find an initial mapping for a given swap strategy. Perform a binary search
        over the number of swap layers, and for each number of swap layers solve a
        subgraph isomorphism problem formulated as a SAT problem.

        Args:
            program_graph (nx.Graph): The program graph with commuting gates, where
                                        each edge represents a two-qubit gate.
            swap_strategy (SwapStrategy): The swap strategy to use to find the initial mapping.
            min_layers (int): The minimum number of swap layers to consider. Defaults to
            the maximum degree of the program graph - 2.
            max_layers (int): The maximum number of swap layers to consider. Defaults to
            the number of qubits in the swap strategy - 2.

        Returns:
            dict[int, SATResult]: A dictionary containing the results of the SAT solver for
                                    each number of swap layers.
        """
        # pylint: disable=too-many-locals
        num_nodes_g1 = len(program_graph.nodes)
        num_nodes_g2 = swap_strategy.distance_matrix.shape[0]
        if num_nodes_g1 > num_nodes_g2:
            return SATResult(False, [], [], 0)
        if min_layers is None:
            # use the maximum degree of the program graph - 2 as the lower bound.
            min_layers = max((d for _, d in program_graph.degree)) - 2
        if max_layers is None:
            max_layers = num_nodes_g2 - 1

        variable_pool = IDPool(start_from=1)
        variables = np.array(
            [
                [variable_pool.id(f"v_{i}_{j}") for j in range(num_nodes_g2)]
                for i in range(num_nodes_g1)
            ],
            dtype=int,
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(variables)}
        binary_search_results = {}

        def interrupt(solver):
            # This function is called to interrupt the solver when the timeout is reached.
            solver.interrupt()

        # Make a cnf for the one-to-one mapping constraint
        cnf1 = []
        for i in range(num_nodes_g1):
            clause = variables[i, :].tolist()
            cnf1.append(clause)
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])
        for j in range(num_nodes_g2):
            clause = variables[:, j].tolist()
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])

        # Perform a binary search over the number of swap layers to find the minimum
        # number of swap layers that satisfies the subgraph isomorphism problem.
        while min_layers < max_layers:
            num_layers = (min_layers + max_layers) // 2

            # Create the connectivity matrix. Note that if the swap strategy cannot reach
            # full connectivity then its distance matrix will have entries with -1. These
            # entries must be treated as False.
            d_matrix = swap_strategy.distance_matrix
            connectivity_matrix = ((-1 < d_matrix) & (d_matrix <= num_layers)).astype(int)
            # Make a cnf for the adjacency constraint
            cnf2 = []
            for e_0, e_1 in program_graph.edges:
                clause_matrix = np.multiply(connectivity_matrix, variables[e_1, :])
                clause = np.concatenate(
                    (
                        [[-variables[e_0, i]] for i in range(num_nodes_g2)],
                        clause_matrix,
                    ),
                    axis=1,
                )
                # Remove 0s from each clause
                cnf2.extend([c[c != 0].tolist() for c in clause])

            cnf = CNF(from_clauses=cnf1 + cnf2)

            with Solver(bootstrap_with=cnf, use_timer=True) as solver:
                # Solve the SAT problem with a timeout.
                # Timer is used to interrupt the solver when the timeout is reached.
                timer = Timer(self.timeout, interrupt, [solver])
                timer.start()
                status = solver.solve_limited(expect_interrupt=True)
                timer.cancel()
                # Get the solution and the elapsed time.
                sol = solver.get_model()
                e_time = solver.time()

                if status:
                    # If the SAT problem is satisfiable, convert the solution to a mapping.
                    mapping = [vid2mapping[idx] for idx in sol if idx > 0]
                    binary_search_results[num_layers] = SATResult(status, sol, mapping, e_time)
                    max_layers = num_layers
                else:
                    # If the SAT problem is unsatisfiable, return the last satisfiable solution.
                    binary_search_results[num_layers] = SATResult(status, sol, [], e_time)
                    min_layers = num_layers + 1

        return binary_search_results

    def remap_graph_with_sat(
        self,
        graph: Union[nx.Graph | SparsePauliOp],
        swap_strategy: SwapStrategy,
    ) -> tuple[int, dict, list] | tuple[None, None, None]:
        """Applies the SAT mapping.

        Args:
            graph: The graph to remap. If a cost operator is provided then it will
                internally be converted to a graph.
            swap_strategy: The swap strategy to use to find the initial mapping.

        Returns:
            tuple: A tuple containing the remapped graph, the edge map, and the number of layers of
            the swap strategy that was used to find the initial mapping. If no solution is found
            then the tuple contains None for each element.
            Note the returned edge map `{k: v}` means that node `k` in the original
            graph gets mapped to node `v` in the Pauli strings.
        """
        op_input = isinstance(graph, SparsePauliOp)

        if op_input:
            graph = self.op2graph(graph)

        num_nodes = len(graph.nodes)
        results = self.find_initial_mappings(graph, swap_strategy, 0, num_nodes - 1)
        solutions = [k for k, v in results.items() if v.satisfiable]

        if len(solutions):
            min_k = min(solutions)
            edge_map = dict(results[min_k].mapping)
            remapped_graph = nx.relabel_nodes(graph, edge_map)

            if op_input:
                return self.graph2op(remapped_graph), edge_map, min_k

            return remapped_graph, edge_map, min_k
        else:
            return None, None, None

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
                edges.append((edge[0], edge[0], np.real(weight)))
            elif len(edge) == 2:
                edges.append((edge[0], edge[1], np.real(weight)))
            else:
                raise ValueError(f"The operator {operator} is not Quadratic.")

        graph.add_weighted_edges_from(edges)

        return graph
