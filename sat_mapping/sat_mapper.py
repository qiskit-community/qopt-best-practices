from __future__ import annotations

from typing import Tuple
import logging
from dataclasses import dataclass
from itertools import combinations
from threading import Timer

import networkx as nx
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

logger = logging.getLogger(__name__)

@dataclass
class SATResult:
    """A data class to hold the result of a SAT solver."""

    satisfiable: bool  # Satisfiable is True if the SAT model could be solved in a given time.
    solution: dict  # The solution to the SAT problem if it is satisfiable.
    mapping: list  # The mapping of nodes in the pattern graph to nodes in the target graph.
    elapsed_time: float  # The time it took to solve the SAT model.


class SATMapper:
    r"""A class to introduce a SAT-approach to solve the subgraph isomorphism problem as well as
    the initial mapping problem in SWAP gate insertion for commuting gates.


    find a mapping for commuting gates using a SAT solver.

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
        start: int | None = None,
        last: int | None = None,
    ) -> dict[int, SATResult]:
        r"""Find an initial mapping for a given swap strategy. Perform a binary search over the number
        of swap layers, and for each number of swap layers solve a subgraph isomorphism problem
        formulated as a SAT problem.

        Args:
            program_graph (nx.Graph): The program graph with commuting gates, where each edge represents a two-qubit gate.
            swap_strategy (SwapStrategy): The swap strategy to use to find the initial mapping.
            start (int): The minimum number of swap layers to consider. Defaults to the maximum degree of the
            program graph - 2.
            last (int): The maximum number of swap layers to consider. Defaults to the number of qubits
            in the swap strategy - 2.

        Returns:
            dict[int, SATResult]: A dictionary containing the results of the SAT solver for each number of swap layers.
        """
        # pylint: disable=too-many-locals
        num_nodes_G1 = len(program_graph.nodes)
        num_nodes_G2 = swap_strategy.distance_matrix.shape[0]
        if num_nodes_G1 > num_nodes_G2:
            return SATResult(False, [], [], 0)
        if start is None:
            # use the maximum degree of the program graph - 2 as the lower bound.
            start = max([d for _, d in program_graph.degree]) - 2
        if last is None:
            last = num_nodes_G2 - 2

        variable_pool = IDPool(start_from=1)
        x = np.array(
            [
                [variable_pool.id(f"v_{i}_{j}") for j in range(num_nodes_G2)]
                for i in range(num_nodes_G1)
            ],
            dtype=int,
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(x)}
        binary_search_results = {}

        def interrupt(solver):
            # This function is called to interrupt the solver when the timeout is reached.
            solver.interrupt()

        # Make a cnf for the one-to-one mapping constraint
        cnf1 = []
        for i in range(num_nodes_G1):
            clause = x[i, :].tolist()
            cnf1.append(clause)
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])
        for j in range(num_nodes_G2):
            clause = x[:, j].tolist()
            for k, m in combinations(clause, 2):
                cnf1.append([-1 * k, -1 * m])

        # Perform a binary search over the number of swap layers to find the minimum number of swap layers
        # that satisfies the subgraph isomorphism problem.
        while start < last:
            num_layers = (start + last) // 2
            distance_matrix = swap_strategy.distance_matrix
            connectivity_matrix = (distance_matrix <= num_layers).astype(int)
            # Make a cnf for the adjacency constraint
            cnf2 = []
            # adj_matrix_G2 = nx.to_numpy_array(G2, dtype=int)
            for e0, e1 in program_graph.edges:
                clause_matrix = np.multiply(connectivity_matrix, x[e1, :])
                clause = np.concatenate(
                    ([[-x[e0, i]] for i in range(num_nodes_G2)], clause_matrix), axis=1
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
                    binary_search_results[num_layers] = SATResult(
                        status, sol, mapping, e_time
                    )
                    last = num_layers
                else:
                    # If the SAT problem is unsatisfiable, return the last satisfiable solution.
                    binary_search_results[num_layers] = SATResult(
                        status, sol, [], e_time
                    )
                    start = num_layers + 1
        return binary_search_results

    def remap_graph_with_sat(self, graph: nx.Graph, swap_strategy) -> Tuple[int, dict, list]:
        """Applies the SAT mapping.

        Note the returned edge map `{k: v}` means that node `k` in the original
        graph gets mapped to node `v` in the Pauli strings.
        """
        n = len(graph.nodes)
        results = self.find_initial_mappings(graph, swap_strategy, 0, n - 2)

        min_k = min([k for k in results.keys() if results[k].satisfiable])
        edge_map = {k: v for k, v in results[min_k].mapping}

        paulis = []

        for edge in graph.edges():
            zop = ["I"] * n
            remapped_edge = (edge_map[edge[0]], edge_map[edge[1]])
            zop[remapped_edge[0]] = "Z"
            zop[remapped_edge[1]] = "Z"
            paulis.append(("".join(zop), 1.0))

        return min_k, edge_map, paulis
