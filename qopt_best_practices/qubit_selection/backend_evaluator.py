from __future__ import annotations
from collections.abc import Callable

from qiskit.transpiler import CouplingMap

from .metric_evaluators import evaluate_fidelity
from .qubit_subset_finders import find_lines

class BackendEvaluator():
    """
    Finds best subset of qubits for a given device that maximizes a given metric for a given geometry.
    This subset can be provided as an initial_layout for the SwapStrategy transpiler pass.
    """

    def __init__(self, backend):

        self.backend = backend
        self.coupling_map = CouplingMap(backend.configuration().coupling_map)

    def evaluate(self, num_qubits, subset_finder: Callable | None = None, metric_eval: Callable | None = None):

        if metric_eval is None:
            metric_eval = evaluate_fidelity

        if subset_finder is None:
            subset_finder = find_lines

        # TODO: add callbacks
        qubit_subsets = subset_finder(num_qubits, self.backend, self.coupling_map)

         # evaluating the subsets
        scores = [metric_eval(subset, self.backend, self.coupling_map.get_edges()) for subset in qubit_subsets]

        # Return the best subset sorted by score (minimize score)
        return min(zip(qubit_subsets, scores), key=lambda x: -x[1])[0], len(qubit_subsets)
