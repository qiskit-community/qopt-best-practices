"""Backend Evaluator"""

from __future__ import annotations
from collections.abc import Callable

from qiskit.transpiler import CouplingMap
from qiskit.providers import Backend

from .metric_evaluators import evaluate_fidelity
from .qubit_subset_finders import find_lines


class BackendEvaluator:
    """
    Finds best subset of qubits for a given device that maximizes a given
    metric for a given geometry.
    This subset can be provided as an initial_layout for the SwapStrategy
    transpiler pass.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.coupling_map = CouplingMap(backend.coupling_map)
        if not self.coupling_map.is_symmetric:
            self.coupling_map.make_symmetric()

    def evaluate(
        self,
        num_qubits: int,
        subset_finder: Callable | None = None,
        metric_eval: Callable | None = None,
    ):
        """
        Args:
            num_qubits: the number of qubits
            subset_finder: callable, will default to "find_line"
            metric_eval: callable, will default to "evaluate_fidelity"
        """

        if metric_eval is None:
            metric_eval = evaluate_fidelity

        if subset_finder is None:
            subset_finder = find_lines

        # TODO: add callbacks
        qubit_subsets = subset_finder(num_qubits, self.backend)

        # evaluating the subsets
        scores = [
            metric_eval(subset, self.backend, self.coupling_map.get_edges())
            for subset in qubit_subsets
        ]

        # Return the best subset sorted by score
        best_subset, best_score = min(zip(qubit_subsets, scores), key=lambda x: -x[1])
        num_subsets = len(qubit_subsets)

        return best_subset, best_score, num_subsets
