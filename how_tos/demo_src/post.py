import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def sample_most_likely(state_vector, num_bits):
    values = list(state_vector.values())
    most_likely = np.argmax(np.abs(values))
    most_likely_bitstring = to_bitstring(most_likely, num_bits)
    most_likely_bitstring.reverse()
    return np.asarray(most_likely_bitstring)

# auxiliary function to plot graphs
def plot_result(G, x):
    colors = ["r" if i == 0 else "b" for i in x]
    pos, default_axes = nx.spring_layout(G), plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, pos=pos)