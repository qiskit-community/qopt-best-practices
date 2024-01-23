"""Subset finders. Currently contains reference implementation
to find lines."""

from __future__ import annotations
import numpy as np
import rustworkx as rx

from qiskit.transpiler import CouplingMap
from qiskit.providers import Backend

def find_lines(length: int, backend: Backend) -> list[int]:
    """Finds all possible lines of length `length` for a specific backend topology.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        The found paths.
    """

    coupling_map = CouplingMap(backend.coupling_map)
    if not coupling_map.is_symmetric:
        coupling_map.make_symmetric()

    all_paths = rx.all_pairs_all_simple_paths(
        coupling_map.graph,
        min_depth=length,
        cutoff=length,
    ).values()

    paths = np.asarray(
        [
            (list(c), list(sorted(list(c))))
            for a in iter(all_paths)
            for b in iter(a)
            for c in iter(a[b])
        ]
    )

    # filter out duplicated paths
    _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)
    paths = paths[:, 0][unique_indices].tolist()

    return paths
