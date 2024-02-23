from collections import defaultdict
import json
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from qiskit_optimization.applications import Maxcut


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


# auxiliary function to convert bit-strings to objective values
def samples_to_objective_values(samples, qp):
    """Convert the samples to values of the objective function."""
    objective_values = defaultdict(float)
    for bit_str, prob in samples.items():

        # Qiskit use little endian hence the [::-1]
        candidate_sol = [int(bit) for bit in bit_str[::-1]]
        fval = qp.objective.evaluate(candidate_sol)
        objective_values[fval] += prob

    return objective_values


# auxiliary function to load saved samples
def load_data(qp):
    depth_one_heron, depth_zero_heron, depth_one_eagle, depth_zero_eagle = {}, {}, {}, {}
    for file in os.listdir("sampler_data/"):
        with open(f"sampler_data/{file}", "r") as fin:
            data = json.load(fin)

        if file.startswith("heron"):
            depth_one_heron.update(data["depth-one"])
            depth_zero_heron.update(data["depth-zero"])
        else:
            depth_one_eagle.update(data["depth-one"])
            depth_zero_eagle.update(data["depth-zero"])

    depth_zero_heron = samples_to_objective_values(depth_zero_heron, qp)
    depth_one_heron = samples_to_objective_values(depth_one_heron, qp)
    depth_zero_eagle = samples_to_objective_values(depth_zero_eagle, qp)
    depth_one_eagle = samples_to_objective_values(depth_one_eagle, qp)

    return depth_one_heron, depth_zero_heron, depth_one_eagle, depth_zero_eagle


# auxiliary function to load the QP from the saved Paulis
def load_qp():

    # First, load the Paulis that encode the MaxCut problem.
    with open("data/125node_example_ising.txt", "r") as fin:
        paulis, lines = [], fin.read()
        for edge in lines.split("\n"):
            try:
                pauli, coefficient = edge.split(", ")
                paulis.append((pauli, float(coefficient)))
            except ValueError:
                pass

    # Next, convert the Paulis to a weighted graph.
    wedges = []
    for pauli_str, coefficient in paulis:
        wedges.append([idx for idx, char in enumerate(pauli_str[::-1]) if char == "Z"] + [{"weight": coefficient}])

    weighted_graph = nx.DiGraph(wedges)

    # Create the Quadratic program to return form the weighted graph.
    mc = Maxcut(weighted_graph)
    qp = mc.to_quadratic_program()

    # Finding the min and max requires CPLEX. If this is not installed we use
    # hard coded values for the sake of the demo.
    try:
        from qiskit_optimization.algorithms import CplexOptimizer
        from qiskit_optimization.problems.quadratic_objective import ObjSense

        # Mximization gives the max cut
        sol = CplexOptimizer().solve(qp)

        # Minimization gives the min cut
        qp2 = mc.to_quadratic_program()
        qp2.objective._sense = ObjSense.MINIMIZE
        sol2 = CplexOptimizer().solve(qp2)

        max_cut, min_cut = sol.fval, sol2.fval
    except:
        max_cut, min_cut = 67, -63

    return qp, max_cut, min_cut


# auxiliary function to help plot cumulative distribution functions
def plot_cdf(objective_values: dict, ax, label):
    x_vals = sorted(objective_values.keys())
    y_vals = np.cumsum([objective_values[x] for x in x_vals])
    ax.plot(x_vals, y_vals, label=label)
