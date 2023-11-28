import json
import os
from typing import Dict


def load_edge_coloring(device: str = "eagle", symmetric: bool = True) -> Dict[tuple, int]:
    """Load an edge coloring.

    Args:
        device: The type of device to use. For now, the supported device in `eagle` only.
        symmetric: A boolean. If this is set to True then if edge `(i, j)` is in the coloring
            we will also add edge `(j, i)`.

    Returns:
        An edge coloring of the coupling map of the given type of device. The edge coloring
        is a mapping between an edge, specified as a tuple, and a number.
    """

    edge_file = os.path.join(os.path.dirname(__file__), f"../data/edge_coloring/{device}.json")

    with open(edge_file, "r") as fin:
        data = json.load(fin)

    edge_coloring = {}
    for edge_data in data["edge colors"]:
        edge = tuple(edge_data["key"])
        edge_coloring[edge] = edge_data["value"]

        if symmetric:
            edge_coloring[edge[::-1]] = edge_data["value"]

    return edge_coloring
