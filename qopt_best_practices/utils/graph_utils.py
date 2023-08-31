"""Graph utils"""

import networkx as nx


def build_graph(paulis: list[tuple[str, float]]) -> nx.Graph:
    """Create a graph by parsing the pauli strings.

    Args:
        paulis: A list of Paulis given as tuple of Pauli string and
            coefficient. E.g., `[("IZZI", 1.0), ("ZIZI", 1.0)]`. Each
            pauli is guaranteed to have two Z's.

    Returns:
        A networkx graph.
    """
    edges = []
    for pauli_str, _ in paulis:
        edges.append([idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"])

    return nx.from_edgelist(edges)

def build_paulis(graph: nx.Graph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_graph`
    """
    pauli_list = []
    for edge in graph.edges():
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        pauli_list.append(("".join(paulis)[::-1], 1.0))

    return pauli_list
