"""A class to solve the SWAP gate insertion initial mapping problem
using simulated annealing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import random
import math
import networkx as nx

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from .initial_mapping import InitialMapping, InitialMappingResult


@dataclass(init=False)
class SAResult(InitialMappingResult):
    """A data class to hold the result of a simulated annealing run."""

    def __init__(
        self,
        mapping: dict,
        cost: float,
        elapsed_time: float,
        metadata: dict | None = None,
    ):
        """Initialize a simulated annealing mapping result."""
        if metadata is None:
            metadata = {}

        super().__init__(
            mapping=mapping,
            objective_value=cost,
            objective_name="cost",
            elapsed_time=elapsed_time,
            metadata={"cost": cost, **metadata},
        )


class SAMapper(InitialMapping):
    r"""Solve the initial qubit mapping problem for commuting 2q-gate blocks
    using simulated annealing.

    Given a program graph :math:`P` (nodes = logical qubits, edges = 2q gates)
    and a swap strategy that defines the hardware connectivity, this class
    finds a mapping from logical qubits to physical qubits that minimises the
    number of program edges that are *not* natively adjacent on the hardware
    (i.e. edges that would otherwise require SWAP gates).

    The class implements the shared :class:`InitialMapping` API so it can be
    used interchangeably with :class:`SATMapper`.
    """

    def __init__(
        self,
        initial_temp: float = 0.01,
        cooling_rate: float = 0.9999,
        stop_temp: float = 1e-8,
        max_iter: int = 10000,
        max_restarts: int = 5,
        verbose: bool = False,
    ):
        """Initialize the SimulatedAnnealingMapper.

        Args:
            initial_temp: Starting temperature for the annealing schedule.
            cooling_rate: Multiplicative cooling factor applied each iteration.
            stop_temp: Temperature below which the annealing loop terminates.
            max_iter: Maximum number of iterations per reheat cycle.
            max_restarts: Number of times the initial solution is reset to
                ``initial_temp`` before returning the best solution found.
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stop_temp = stop_temp
        self.max_iter = max_iter
        self.max_restarts = max_restarts
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_initial_mapping(
        self,
        program_graph: nx.Graph,
        swap_strategy: SwapStrategy,
    ) -> SAResult:
        """Find an initial mapping using simulated annealing.

        Args:
            program_graph: The program graph where each node is a logical
                qubit and each edge represents a 2q gate.
            swap_strategy: Defines the hardware topology via its
                ``distance_matrix``.  The number of hardware qubits is
                inferred from the shape of the distance matrix.

        Returns:
            SAResult: The best mapping, cost, and elapsed time found by the
                annealing run.
        """
        t_start = time.time()

        n_physical = swap_strategy.distance_matrix.shape[0]
        n_logical = program_graph.number_of_nodes()

        if n_logical > n_physical:
            raise ValueError(
                f"Program graph has {n_logical} nodes but the swap strategy "
                f"only defines {n_physical} physical qubits."
            )

        # Generate all 2q-gate layers for the full swap strategy on the
        # hardware line.  The SA searches for the initial mapping that
        # maximises natively-connected edges across all swap layers.
        list_2q = SWAP_pairs(n_physical)

        # Pad the program graph with isolated logical nodes so the annealer
        # can assign a subset of the physical qubits when n_physical > n_logical.
        padded_program_graph = program_graph.copy()
        padded_program_graph.add_nodes_from(range(n_physical))

        # Internal mapping is {physical_qubit: logical_qubit}. Start from the
        # identity placement over the padded logical register.
        initial_mapping = {i: i for i in range(n_physical)}

        result = simulated_annealing_func(
            G_original=padded_program_graph,
            initial_mapping=initial_mapping,
            list_2q=list_2q,
            initial_temp=self.initial_temp,
            cooling_rate=self.cooling_rate,
            stop_temp=self.stop_temp,
            max_iter=self.max_iter,
            verbose=self.verbose,
            max_restarts=self.max_restarts,
        )

        result.elapsed_time = time.time() - t_start
        return result

    def remap_graph_with_sa(
        self,
        graph: nx.Graph | SparsePauliOp,
        swap_strategy: SwapStrategy,
    ) -> tuple[nx.Graph | SparsePauliOp, dict, InitialMappingResult] | tuple[None, None, None]:
        """Apply the simulated annealing mapping.

        Args:
            graph: The program graph to remap.  A :class:`SparsePauliOp` is
                accepted and converted to a graph internally.
            swap_strategy: The swap strategy used to determine hardware
                connectivity.

        Returns:
            A 3-tuple ``(remapped_graph, edge_map, result)`` where

            * ``remapped_graph`` – graph with nodes relabelled to physical
              qubit indices,
            * ``edge_map`` – ``{logical_qubit: physical_qubit}`` mapping,
            * ``result`` – common :class:`InitialMappingResult` with the best
              cost returned by the annealer as its objective value.

            If the mapping fails (e.g. too few physical qubits), returns
            ``(None, None, None)``.

        Note:
            The returned ``edge_map`` ``{k: v}`` means that node ``k`` in the
            original graph gets mapped to physical qubit ``v``.
        """
        return self.remap_graph(graph, swap_strategy)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hardware_graph_from_strategy(swap_strategy: SwapStrategy) -> nx.Graph:
        """Build the base hardware coupling graph (layer-0 connectivity) from
        a :class:`SwapStrategy`.

        Edges are added between any two qubits whose entry in the strategy's
        ``distance_matrix`` equals 1 (directly connected, no swaps needed).
        """
        d_matrix = swap_strategy.distance_matrix
        n = d_matrix.shape[0]
        hardware_graph = nx.Graph()
        hardware_graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if d_matrix[i][j] == 1:
                    hardware_graph.add_edge(i, j)
        return hardware_graph


class SimulatedAnnealingMapper(SAMapper):
    """Backward-compatible name for :class:`SAMapper`."""


def SWAP_pairs(nq):
    qubit_order = list(range(nq))
    list_2q = [[(qubit_order[ii], qubit_order[ii + 1]) for ii in range(0, nq - 1, 2)]]
    for i in range(0, nq):
        for j in range(i % 2, nq - 1, 2):
            qubit_order[j], qubit_order[j + 1] = qubit_order[j + 1], qubit_order[j]
        list_2q.append(
            [tuple([qubit_order[ii], qubit_order[ii + 1]]) for ii in range(i % 2, nq - 1, 2)]
        )
    return list_2q


def simulated_annealing_func(
    G_original: nx.Graph,
    initial_mapping: dict,
    list_2q: list,
    initial_temp=0.01,
    cooling_rate=0.9999,
    stop_temp=1e-8,
    max_iter=10000,
    verbose=False,
    max_restarts=5,
) -> SAResult:
    n = G_original.number_of_nodes()
    max_iter = int(max_iter)

    # Use list for O(1) index access (faster than dict)
    mapping = [initial_mapping[k] for k in range(n)]

    # Adjacency matrix for O(1) edge lookup (avoids tuple creation + hashing)
    adj = [[False] * n for _ in range(n)]
    for u, v in G_original.edges():
        adj[u][v] = True
        adj[v][u] = True

    num_layers = len(list_2q)

    # Precompute: for each node, list of (layer_index, partner_node) for all 2q edges
    node_edges = [[] for _ in range(n)]
    for l_idx in range(num_layers):
        for edge in list_2q[l_idx]:
            if len(edge) == 2:
                u, v = edge
                node_edges[u].append((l_idx, v))
                node_edges[v].append((l_idx, u))

    # Per-layer valid and invalid 2q-edge counts
    valid_count = [0] * num_layers
    invalid_count = [0] * num_layers

    def recompute_layer_counts():
        for l_idx in range(num_layers):
            vc = 0
            ic = 0
            for edge in list_2q[l_idx]:
                if len(edge) == 2:
                    u, v = edge
                    if adj[mapping[u]][mapping[v]]:
                        vc += 1
                    else:
                        ic += 1
            valid_count[l_idx] = vc
            invalid_count[l_idx] = ic

    recompute_layer_counts()

    def compute_cost_depth():
        cnots = 0
        depth = 0
        for l in range(num_layers - 1, -1, -1):
            cnots -= invalid_count[l]
            if valid_count[l] > 0:
                return cnots, depth
            depth -= 1
        return cnots, depth

    current_cost, current_depth = compute_cost_depth()
    best_mapping = mapping[:]
    best_cost = current_cost
    best_depth = current_depth

    T = initial_temp
    iteration = 0
    trace: dict = {"cost": [], "depth": [], "iterations": [], "T": []}

    # Local references for hot-path speedup
    _randint = random.randint
    _random = random.random
    _exp = math.exp
    n_minus_1 = n - 1
    n_minus_2 = n - 2
    best_overall_cost = best_cost
    best_overall_mapping = best_mapping[:]

    for _ in range(max_restarts):
        T = initial_temp
        iteration = 0
        current_iter = trace["iterations"][-1] if trace["iterations"] else 0
        mapping = [initial_mapping[k] for k in range(n)]
        recompute_layer_counts()
        current_cost, current_depth = compute_cost_depth()
        best_mapping = mapping[:]
        best_cost = current_cost
        best_depth = current_depth

        while T > stop_temp and iteration < max_iter:
            # Fast random pair selection (avoids range() + sample())
            i = _randint(0, n_minus_1)
            j = _randint(0, n_minus_2)
            if j >= i:
                j += 1

            phys_i = mapping[i]
            phys_j = mapping[j]

            # Incremental cost: only check edges involving swapped nodes i or j
            changes = []

            for l_idx, partner in node_edges[i]:
                if partner == j:
                    continue  # edge (i,j): swapping both endpoints doesn't change validity
                mp = mapping[partner]
                old_v = adj[phys_i][mp]
                new_v = adj[phys_j][mp]
                if old_v != new_v:
                    changes.append((l_idx, 1 if new_v else -1))

            for l_idx, partner in node_edges[j]:
                if partner == i:
                    continue
                mp = mapping[partner]
                old_v = adj[phys_j][mp]
                new_v = adj[phys_i][mp]
                if old_v != new_v:
                    changes.append((l_idx, 1 if new_v else -1))

            # Apply incremental changes
            for l_idx, delta in changes:
                valid_count[l_idx] += delta
                invalid_count[l_idx] -= delta

            neighbor_cost, neighbor_depth = compute_cost_depth()
            delta_cost = neighbor_cost - current_cost

            # Accept or reject the new solution
            if delta_cost < 0 or _random() < _exp(-delta_cost / T):
                mapping[i] = phys_j
                mapping[j] = phys_i
                current_cost = neighbor_cost
                current_depth = neighbor_depth

                if current_cost < best_cost:
                    best_mapping = mapping[:]
                    best_cost = current_cost
                    best_depth = current_depth

                    if verbose:
                        print(
                            f"best depth:{best_depth} | best cost:{best_cost} | iteration:{iteration} | T:{T:.4f}"
                        )

                    trace["cost"].append(best_cost)
                    trace["depth"].append(best_depth)
                    trace["iterations"].append(current_iter + iteration)
                    trace["T"].append(T)

            else:
                # Reject: revert incremental changes
                for l_idx, delta in changes:
                    valid_count[l_idx] -= delta
                    invalid_count[l_idx] += delta

            T *= cooling_rate
            iteration += 1

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_mapping = best_mapping[:]

    best_mapping = dict(enumerate(best_overall_mapping))

    best_mapping = {v: k for k, v in best_mapping.items()}
    return SAResult(
        mapping=best_mapping,
        cost=best_overall_cost,
        elapsed_time=0.0,
        metadata={"trace": trace},
    )
