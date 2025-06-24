"""QAOA Cost function utils"""

from collections import defaultdict
import networkx as nx
import numpy as np

from qiskit.quantum_info import SparsePauliOp

_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)


def evaluate_sparse_pauli(state: int, observable: SparsePauliOp) -> complex:
    """Utility for the evaluation of the expectation value of a measured state."""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def qaoa_sampler_cost_fun(params, ansatz, hamiltonian, sampler):
    """Standard sampler-based QAOA cost function to be plugged into optimizer routines."""
    job = sampler.run(ansatz, params)
    sampler_result = job.result()
    sampled = sampler_result.quasi_dists[0]

    # a dictionary containing: {state: (measurement probability, value)}
    evaluated = {
        state: (probability, evaluate_sparse_pauli(state, hamiltonian))
        for state, probability in sampled.items()
    }

    result = sum(probability * value for probability, value in evaluated.values())

    return result


def counts_to_maxcut_cost(graph: nx.Graph, counts: dict):
    """Convert a dict of counts to a dict of cut values for MaxCut.
    
    This method computes the cost function x^T Q x.
    """
    nshots = sum(counts.values())
    adj_mat = nx.adjacency_matrix(graph, nodelist=range(graph.order())).toarray()
    cost_vals = defaultdict(float)

    for bit_str, count in counts.items():
        x = np.array([int(x) for x in bit_str[::-1]])
        val = float(x.T @ adj_mat @ (1-x))
        cost_vals[val] += count / nshots

    return cost_vals
