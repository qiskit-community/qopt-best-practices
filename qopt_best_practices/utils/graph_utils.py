"""Graph utils"""

import networkx as nx


def build_max_cut_graph(paulis: list[tuple[str, float]]) -> nx.Graph:
    """Create a graph by parsing the pauli strings.

    Args:
        paulis: A list of Paulis given as tuple of Pauli string and
            coefficient. E.g., `[("IZZI", 1.0), ("ZIZI", 1.0)]`. Each
            pauli is guaranteed to have two Z's.

    Returns:
        A networkx graph.
    """
    graph = nx.Graph()

    for pauli_str, weight in paulis:
        edge = [idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"]
        graph.add_edge(edge[0], edge[1], weight=weight)

    return graph


def build_max_cut_paulis(graph: nx.Graph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in graph.edges():
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1]).get("weight", 1.0)

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list
