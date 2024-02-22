import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def sample_most_likely(state_vector, num_bits):
    keys = list(state_vector.keys())
    values = list(state_vector.values())
    most_likely = keys[np.argmax(np.abs(values))]
    most_likely_bitstring = to_bitstring(most_likely, num_bits)
    most_likely_bitstring.reverse()
    return np.asarray(most_likely_bitstring)

# auxiliary function to plot graphs
def plot_result(G, x):
    colors = ['tab:grey' if i == 0 else 'tab:purple' for i in x]
    pos, default_axes = nx.spring_layout(G), plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, pos=pos)


def plot_distribution(final_distribution):
    matplotlib.rcParams.update({'font.size': 10})
    final_bits = final_distribution.binary_probabilities()
    position = np.argmax(np.abs(list(final_bits.values())))
    
    fig = plt.figure(figsize = (11,6))
    ax=fig.add_subplot(1,1,1)
    plt.xticks(rotation=45)
    plt.title("Result Distribution")
    plt.xlabel("Bitstrings (reversed)")
    plt.ylabel("Probability")
    ax.bar(list(final_bits.keys()), list(final_bits.values()), color='tab:grey')
    ax.get_children()[position].set_color('tab:purple') 
    plt.show()