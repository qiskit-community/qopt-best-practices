"""Subset finders. Currently contains reference implementation
to find lines."""

from __future__ import annotations
import itertools

import rustworkx as rx

from qiskit.transpiler import CouplingMap

# TODO: backend typehint. Currently, only BackendV1 is supported
#       Might make sense to extend to BackendV2 for generality
def find_lines(
    length: int, backend, coupling_map: CouplingMap | None = None
) -> list[int]:
    """Finds all possible lines of length `length` for a specific backend topology.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        The found paths.
    """

    # how expensive is it to get the coupling map from the backend?
    # might make sense to make backend the only input for simplicity
    if coupling_map is None:
        coupling_map = CouplingMap(backend.configuration().coupling_map)
    size = coupling_map.size()

    # picking the lines
    paths = list(itertools.chain.from_iterable(
        rx.all_pairs_all_simple_paths(
            coupling_map.graph,
            min_depth=length,
            cutoff=length,
        ).values()
    ))

    return paths
