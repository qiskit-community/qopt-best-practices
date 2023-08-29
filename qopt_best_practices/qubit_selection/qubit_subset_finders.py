from __future__ import annotations
import rustworkx as rx

from qiskit.transpiler import CouplingMap

# TODO: backend typehint. Currently, only BackendV1 is supported
#       Might make sense to extend to BackendV2 for generality

def find_lines(length: int, backend, coupling_map: CouplingMap | None = None) -> list[int]:
    """Finds all possible lines of lengt `length` for a specific backend topology.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        The found paths.
    """

    # how expensive is it to get the coupling map from the backend?
    # might make sense to make backend the only input for simplicity
    if coupling_map is None:
        coupling_map = CouplingMap(backend.configuration().coupling_map)
    paths, size = [], coupling_map.size()

    # picking the lines
    for node1 in range(size):
        for node2 in range(node1 + 1, size):
            paths.extend(
                rx.all_simple_paths(
                    coupling_map.graph,
                    node1,
                    node2,
                    min_depth=length,
                    cutoff=length,
                )
            )

    return paths

