"""A class to solve the SWAP gate insertion initial mapping problem
using the simulated annealing approach from https://arxiv.org/pdf/2505.17944v1.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from .initial_mapping import InitialMapping, InitialMappingResult


@dataclass(init=False)
class SAResult(InitialMappingResult):
    """A data class to hold the result of a simulated annealing run.

    The objective value (``cost``) is the number of swap layers of the swap
    strategy that are needed to execute all two-qubit gates of the program
    graph under the found mapping.
    """

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
    and a swap strategy, this class finds a mapping from logical to physical
    qubits that minimises the number of swap layers needed to execute every
    program edge.  The distance matrix of the swap strategy gives, for any
    pair of physical qubits, the number of swap layers after which they
    become adjacent; the objective is the maximum of this distance over all
    program edges.

    The search anneals a sequence of feasibility problems: given the best
    number of layers :math:`M` found so far, it tries to satisfy "every
    program edge is within :math:`M - 1` layers" by minimising the total
    excess distance.  Whenever the excess reaches zero the level is lowered
    again, mirroring the binary search of :class:`SATMapper`.  Half of the
    proposed moves are targeted repairs that relocate an endpoint of a
    violating edge next to its partner; the rest are uniform random swaps.

    Edge weights of the program graph are ignored: only which pairs of
    qubits interact matters for the swap-layer count.

    The class implements the shared :class:`InitialMapping` API so it can be
    used interchangeably with :class:`SATMapper`.
    """

    def __init__(
        self,
        initial_temp: float | None = None,
        cooling_rate: float | None = None,
        stop_temp: float = 1e-8,
        max_iter: int = 10000,
        max_restarts: int = 5,
        verbose: bool = False,
        seed: int | None = None,
        repair_prob: float = 0.5,
    ):
        """Initialize the SimulatedAnnealingMapper.

        Args:
            initial_temp: Starting temperature of each annealing stage.  If
                ``None`` (default) the temperature is set adaptively from the
                cost deltas of sampled random moves.
            cooling_rate: Multiplicative cooling factor applied each
                iteration.  If ``None`` (default) it is derived so that the
                temperature decays by four orders of magnitude over one
                stage of ``max_iter`` iterations.
            stop_temp: Lower bound on the temperature.
            max_iter: Maximum number of iterations per annealing stage, i.e.
                per attempted swap-layer level.
            max_restarts: Multiplier on ``max_iter`` that fixes the total
                iteration budget (``max_iter * max_restarts``).  When a level
                cannot be satisfied the search restarts from a fresh greedy
                placement until the budget is exhausted.
            verbose: Print progress whenever a better level is reached.
            seed: Optional seed for the internal random number generator.
            repair_prob: Probability of proposing a targeted repair move
                instead of a uniform random swap.
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stop_temp = stop_temp
        self.max_iter = int(max_iter)
        self.max_restarts = int(max_restarts)
        self.verbose = verbose
        self.seed = seed
        self.repair_prob = repair_prob

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
                qubit and each edge represents a 2q gate.  Node labels may
                be arbitrary hashables.
            swap_strategy: Defines the hardware topology and swap layers via
                its ``distance_matrix``.

        Returns:
            SAResult: The best mapping found, with the number of swap layers
                it requires as the objective value.

        Raises:
            ValueError: If the program graph has more nodes than the swap
                strategy has physical qubits, or if the swap strategy cannot
                make all program edges adjacent.
        """
        t_start = time.time()

        dist_np = swap_strategy.distance_matrix
        n_phys = dist_np.shape[0]
        nodes = list(program_graph.nodes())
        n_log = len(nodes)

        if n_log > n_phys:
            raise ValueError(
                f"Program graph has {n_log} nodes but the swap strategy "
                f"only defines {n_phys} physical qubits."
            )

        rng = random.Random(self.seed)

        # Entries of -1 mean the pair never becomes adjacent; encode them as
        # an unreachable distance so such placements are never selected.
        max_finite = int(dist_np[dist_np >= 0].max()) if n_phys > 1 else 0
        unreachable = max_finite + n_phys + 1
        dist_arr = np.where(dist_np >= 0, dist_np, unreachable).astype(np.int64)
        dist = dist_arr.tolist()

        # Work on integer logical indices 0..n_log-1; padded indices up to
        # n_phys-1 are isolated placeholders for unused physical qubits.
        idx = {u: i for i, u in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in program_graph.edges() if u != v]
        nbrs: list[list[int]] = [[] for _ in range(n_phys)]
        inc_edges: list[list[int]] = [[] for _ in range(n_phys)]
        for e_idx, (u, v) in enumerate(edges):
            nbrs[u].append(v)
            nbrs[v].append(u)
            inc_edges[u].append(e_idx)
            inc_edges[v].append(e_idx)

        trace: dict = {"num_swap_layers": [], "iterations": []}

        if not edges:
            mapping = {u: idx[u] for u in nodes}
            return SAResult(
                mapping=mapping,
                cost=0,
                elapsed_time=time.time() - t_start,
                metadata={"num_swap_layers": 0, "trace": trace},
            )

        # For repair moves: physical qubits sorted by distance from q, and
        # how many of them are within each number of layers (vectorised).
        keyed = dist_arr.copy()
        np.fill_diagonal(keyed, unreachable + 1)
        order_np = np.argsort(keyed, axis=1, kind="stable")[:, : n_phys - 1]
        values_np = np.take_along_axis(keyed, order_np, axis=1)
        sorted_by_dist = order_np.tolist()
        prefix_count = [
            np.searchsorted(values_np[q], np.arange(max_finite + 1), side="right").tolist()
            for q in range(n_phys)
        ]
        del keyed, order_np, values_np

        pos = self._greedy_init(nbrs, dist, n_phys, n_log, rng)
        at = [0] * n_phys
        for log, phys in enumerate(pos):
            at[phys] = log

        def cur_max():
            return max(dist[pos[u]][pos[v]] for u, v in edges)

        best_layers = cur_max()
        best_pos = pos[:]
        used = 0
        total_budget = self.max_iter * self.max_restarts
        lower_bound = max(0, max(len(n) for n in nbrs) - 2)

        trace["num_swap_layers"].append(best_layers)
        trace["iterations"].append(used)

        # The target level descends geometrically and the step is bisected
        # when a stage fails, mirroring a binary search over the number of
        # swap layers; one-level steps that still fail trigger a restart.
        gap = max(1, best_layers // 16)

        while best_layers > lower_bound and used < total_budget:
            level = max(best_layers - gap, 0)
            stage_budget = min(self.max_iter, total_budget - used)
            solved, used = self._anneal_level(
                level,
                pos,
                at,
                edges,
                nbrs,
                inc_edges,
                dist,
                sorted_by_dist,
                prefix_count,
                n_phys,
                stage_budget,
                used,
                rng,
            )
            if solved:
                best_layers = cur_max()
                best_pos = pos[:]
                gap = max(1, best_layers // 16)
                trace["num_swap_layers"].append(best_layers)
                trace["iterations"].append(used)
                if self.verbose:
                    print(f"num swap layers: {best_layers} | iteration: {used}")
            else:
                pos[:] = best_pos
                for log, phys in enumerate(pos):
                    at[phys] = log
                if gap > 1:
                    gap = max(1, gap // 2)
                else:
                    # Even a single-level step failed: restart from a fresh
                    # greedy placement (or a random one for diversity).
                    pos[:] = self._greedy_init(nbrs, dist, n_phys, n_log, rng)
                    if rng.random() < 0.5:
                        rng.shuffle(pos)
                    for log, phys in enumerate(pos):
                        at[phys] = log
                    if cur_max() < best_layers:
                        best_layers = cur_max()
                        best_pos = pos[:]
                    gap = max(1, best_layers // 16)

        if best_layers >= unreachable:
            raise ValueError("The swap strategy cannot make all program edges adjacent.")

        mapping = {u: best_pos[idx[u]] for u in nodes}
        return SAResult(
            mapping=mapping,
            cost=best_layers,
            elapsed_time=time.time() - t_start,
            metadata={"num_swap_layers": best_layers, "trace": trace},
        )

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
            * ``result`` – common :class:`InitialMappingResult` with the
              number of swap layers as its objective value.

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
    def _greedy_init(nbrs, dist, n_phys, n_log, rng):
        """Construct an initial placement.

        Nodes are placed in order of decreasing degree, each on the free
        physical qubit that minimises the (max, sum) distance to its already
        placed neighbours.  Ties are broken randomly so repeated calls give
        different placements.
        """
        order = sorted(range(n_log), key=lambda u: -len(nbrs[u]))
        pos = [-1] * n_phys
        free = set(range(n_phys))
        for u in order:
            placed = [w for w in nbrs[u] if pos[w] >= 0]
            if placed:
                q = min(
                    free,
                    key=lambda q, placed=tuple(placed): (
                        max(dist[q][pos[w]] for w in placed),
                        sum(dist[q][pos[w]] for w in placed),
                        rng.random(),
                    ),
                )
            else:
                q = rng.choice(sorted(free))
            pos[u] = q
            free.remove(q)
        free = sorted(free)
        rng.shuffle(free)
        for u in range(n_log, n_phys):
            pos[u] = free.pop()
        return pos

    def _anneal_level(
        self,
        level,
        pos,
        at,
        edges,
        nbrs,
        inc_edges,
        dist,
        sorted_by_dist,
        prefix_count,
        n_phys,
        stage_budget,
        used,
        rng,
    ):
        """Anneal the feasibility problem "every edge within ``level`` layers".

        The stage cost is the total excess distance over ``level`` summed
        over all program edges; it reaches zero exactly when the level is
        satisfied.  ``pos`` and ``at`` are updated in place.  Returns
        ``(solved, used)`` with the updated iteration count.
        """
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        cost = sum(max(0, dist[pos[u]][pos[v]] - level) for u, v in edges)

        # Violating edges with O(1) sampling and removal.
        viol_list = [e for e, (u, v) in enumerate(edges) if dist[pos[u]][pos[v]] > level]
        viol_pos = {e: i for i, e in enumerate(viol_list)}

        def propose():
            """Pick the pair of logical indices to swap, or None to skip."""
            if viol_list and rng.random() < self.repair_prob:
                u, v = edges[viol_list[int(rng.random() * len(viol_list))]]
                if rng.random() < 0.5:
                    u, v = v, u
                # Relocate u onto a physical qubit within `level` layers of v.
                n_cands = prefix_count[pos[v]][min(level, len(prefix_count[pos[v]]) - 1)]
                if n_cands == 0:
                    return None
                target = sorted_by_dist[pos[v]][int(rng.random() * n_cands)]
                if target == pos[u]:
                    return None
                return u, at[target]
            i = rng.randrange(n_phys)
            j = rng.randrange(n_phys - 1)
            if j >= i:
                j += 1
            return i, j

        def swap_delta(i, j):
            pi, pj = pos[i], pos[j]
            delta = 0
            for w in nbrs[i]:
                if w != j:
                    pw = pos[w]
                    delta += max(0, dist[pj][pw] - level) - max(0, dist[pi][pw] - level)
            for w in nbrs[j]:
                if w != i:
                    pw = pos[w]
                    delta += max(0, dist[pi][pw] - level) - max(0, dist[pj][pw] - level)
            return delta

        # Temperature schedule for this stage.
        if self.initial_temp is not None:
            temp = self.initial_temp
        else:
            ups = []
            for _ in range(50):
                move = propose()
                if move is not None:
                    d = swap_delta(*move)
                    if d > 0:
                        ups.append(d)
            ups.sort()
            temp = ups[len(ups) // 2] / math.log(2) if ups else 1.0
        if self.cooling_rate is not None:
            alpha = self.cooling_rate
        else:
            t_end = max(self.stop_temp, temp * 1e-4)
            alpha = (t_end / temp) ** (1.0 / stage_budget)

        _random = rng.random
        _exp = math.exp
        stop_temp = self.stop_temp

        for _ in range(stage_budget):
            used += 1
            if cost == 0:
                return True, used
            move = propose()
            if move is None:
                continue
            i, j = move
            delta = swap_delta(i, j)

            if delta <= 0 or _random() < _exp(-delta / temp):
                pi, pj = pos[i], pos[j]
                pos[i], pos[j] = pj, pi
                at[pj], at[pi] = i, j
                cost += delta
                # Refresh the violation status of every affected edge.
                for e_idx in inc_edges[i]:
                    self._update_violation(e_idx, edges, pos, dist, level, viol_list, viol_pos)
                for e_idx in inc_edges[j]:
                    if e_idx not in inc_edges[i]:
                        self._update_violation(e_idx, edges, pos, dist, level, viol_list, viol_pos)
            temp = max(temp * alpha, stop_temp)

        return cost == 0, used

    @staticmethod
    def _update_violation(e_idx, edges, pos, dist, level, viol_list, viol_pos):
        """Add or remove edge ``e_idx`` from the violation structure."""
        u, v = edges[e_idx]
        violating = dist[pos[u]][pos[v]] > level
        if violating and e_idx not in viol_pos:
            viol_pos[e_idx] = len(viol_list)
            viol_list.append(e_idx)
        elif not violating and e_idx in viol_pos:
            i = viol_pos.pop(e_idx)
            last = viol_list.pop()
            if last != e_idx:
                viol_list[i] = last
                viol_pos[last] = i


class SimulatedAnnealingMapper(SAMapper):
    """Backward-compatible name for :class:`SAMapper`."""


def swap_pairs(nq):
    """Generate the 2q-gate layers of a full line swap strategy on ``nq`` qubits.

    Kept for backward compatibility; :class:`SAMapper` now reads the swap
    layers directly from the distance matrix of the swap strategy.
    """
    qubit_order = list(range(nq))
    list_2q = [[(qubit_order[ii], qubit_order[ii + 1]) for ii in range(0, nq - 1, 2)]]
    for i in range(0, nq):
        for j in range(i % 2, nq - 1, 2):
            qubit_order[j], qubit_order[j + 1] = qubit_order[j + 1], qubit_order[j]
        list_2q.append(
            [tuple([qubit_order[ii], qubit_order[ii + 1]]) for ii in range(i % 2, nq - 1, 2)]
        )
    return list_2q
