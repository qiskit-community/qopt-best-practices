"""Unit tests for annotated transpilation pipeline"""

import unittest
import networkx as nx
from networkx import barabasi_albert_graph

from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    Commuting2qGateRouter,
)
from qiskit.circuit.library import CXGate, qaoa_ansatz
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qiskit.providers.fake_provider import GenericBackendV2

from qopt_best_practices.circuit_library import annotated_qaoa_ansatz
from qopt_best_practices.utils import build_max_cut_paulis
from qopt_best_practices.transpilation.cost_layer import get_cost_layer
from qopt_best_practices.transpilation.prepare_cost_layer import PrepareCostLayer
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping
from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.qubit_selection import BackendEvaluator
from qopt_best_practices.transpilation import (
    AnnotatedPrepareCostLayer,
    AnnotatedCommuting2qGateRouter,
    AnnotatedSwapToFinalMapping,
    SynthesizeAndSimplifyCostLayer,
    UnrollBoxes,
)


def get_problem_barabasi(n=4, m=3):
    """Get problem data for barabasi albert graph"""
    graph = barabasi_albert_graph(n=n, m=m, seed=42)
    local_correlators = build_max_cut_paulis(graph)
    cost_operator = SparsePauliOp.from_list(local_correlators)
    cost_layer = get_cost_layer(cost_operator)
    return cost_operator, cost_layer, graph


def get_problem_maxcut(n=4, elist=None):
    """Get problem data for maxcut graph"""
    if elist is None:
        elist = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(elist)
    local_correlators = build_max_cut_paulis(graph)
    cost_operator = SparsePauliOp.from_list(local_correlators)
    cost_layer = get_cost_layer(cost_operator)
    return cost_operator, cost_layer, graph


def backend_and_layout_a(cost_layer):
    """Get backend and layout for a given cost layer"""
    backend = GenericBackendV2(
        num_qubits=cost_layer.num_qubits,
        basis_gates=["x", "sx", "cz", "id", "rz"],
        seed=0,
    )
    path_finder = BackendEvaluator(backend)
    path, _, _ = path_finder.evaluate(cost_layer.num_qubits)
    initial_layout = Layout.from_intlist(path, cost_layer.qregs[0])
    return backend, initial_layout


class TestAnnotatedTranspilation(unittest.TestCase):
    """Test annotated transpilation pipeline"""

    def setUp(self):
        self.estimator = StatevectorEstimator()
        self.test_cases = [
            (1, 4, 2, [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)]),
            (2, 4, 2, [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)]),
            (3, 4, 2, [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0)]),
            (2, 5, 3, [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]),
            (
                2,
                7,
                3,
                [
                    (0, 1, 1.0),
                    (0, 2, 1.0),
                    (1, 2, 1.0),
                    (2, 3, 1.0),
                    (3, 1, 1.0),
                    (3, 4, 1.0),
                    (3, 5, 1.0),
                    (4, 6, 1.0),
                ],
            ),
        ]

    def _get_swap_strategy(self, circuit):
        swap_strategy = SwapStrategy.from_line(list(range(circuit.num_qubits)))
        edge_coloring = {(i, i + 1): (i + 1) % 2 for i in range(circuit.num_qubits)}
        return swap_strategy, edge_coloring

    def _estimate(self, circuit, hamiltonian, param_values):
        circuit.remove_final_measurements()
        isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
        result = self.estimator.run([(circuit, isa_hamiltonian, param_values)]).result()[0]
        return list(result.data.values())

    def _assert_equivalence(self, expvals_1, expvals_2, circuit_1, circuit_2):
        self.assertAlmostEqual(float(expvals_1[0]), float(expvals_2[0]), places=12)
        for key in circuit_1.count_ops():
            self.assertEqual(circuit_1.count_ops()[key], circuit_2.count_ops()[key])

    def _run_qopt_and_annot(  # pylint: disable=too-many-positional-arguments
        self, cost_layer, hamiltonian, num_qaoa_layers, backend, initial_layout, optimized=False
    ):
        """Run both the previous qopt pipeline and new annotated pipeline"""

        swap_strategy, edge_coloring = self._get_swap_strategy(cost_layer)

        # QOpt pipeline
        qopt_passes = [
            PrepareCostLayer(),
            Commuting2qGateRouter(swap_strategy, edge_coloring),
            SwapToFinalMapping(),
        ]
        if optimized:
            qopt_passes += [
                HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
                InverseCancellation(gates_to_cancel=[CXGate()]),
            ]
        qopt_passes.append(QAOAConstructionPass(num_layers=num_qaoa_layers))

        qopt_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        qopt_pm.pre_init = PassManager(qopt_passes)
        qopt_transpiled = qopt_pm.run(cost_layer)

        # Annotated pipeline
        annotated_ansatz = annotated_qaoa_ansatz(hamiltonian, reps=num_qaoa_layers)
        annot_passes = [
            AnnotatedPrepareCostLayer(),
            AnnotatedCommuting2qGateRouter(swap_strategy, edge_coloring),
            AnnotatedSwapToFinalMapping(),
        ]
        if optimized:
            annot_passes.append(
                SynthesizeAndSimplifyCostLayer(basis_gates=["x", "cx", "sx", "rz", "id"])
            )
        annot_passes.append(UnrollBoxes())

        annot_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        annot_pm.pre_init = PassManager(annot_passes)
        annot_transpiled = annot_pm.run(annotated_ansatz)

        return qopt_transpiled, annot_transpiled

    def _run_standard_and_annot(  # pylint: disable=too-many-positional-arguments
        self, cost_layer, hamiltonian, num_qaoa_layers, backend, initial_layout, optimized=False
    ):
        """Run both the standard qaoa and new annotated pipeline"""

        swap_strategy, edge_coloring = self._get_swap_strategy(cost_layer)

        # Standard pipeline
        standard_ansatz = qaoa_ansatz(hamiltonian, reps=num_qaoa_layers)
        standard_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        standard_transpiled = standard_pm.run(standard_ansatz)

        # Annotated pipeline
        annotated_ansatz = annotated_qaoa_ansatz(hamiltonian, reps=num_qaoa_layers)
        annot_passes = [
            AnnotatedPrepareCostLayer(),
            AnnotatedCommuting2qGateRouter(swap_strategy, edge_coloring),
            AnnotatedSwapToFinalMapping(),
        ]
        if optimized:
            annot_passes.append(
                SynthesizeAndSimplifyCostLayer(basis_gates=["x", "cx", "sx", "rz", "id"])
            )
        annot_passes.append(UnrollBoxes())

        annot_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        annot_pm.pre_init = PassManager(annot_passes)
        annot_transpiled = annot_pm.run(annotated_ansatz)

        return standard_transpiled, annot_transpiled

    def _run_comparison_qopt(  # pylint: disable=too-many-positional-arguments
        self, hamiltonian, cost_layer, num_qaoa_layers, backend, initial_layout, optimized=False
    ):
        """Run both transpilation pipelines and compare expectation values
        and number of final operations"""

        param_values = [5.11350346] * num_qaoa_layers + [5.52673212] * num_qaoa_layers
        qopt_transpiled, annot_transpiled = self._run_qopt_and_annot(
            cost_layer, hamiltonian, num_qaoa_layers, backend, initial_layout, optimized
        )
        eval_qopt = self._estimate(qopt_transpiled, hamiltonian, param_values)
        eval_annot = self._estimate(annot_transpiled, hamiltonian, param_values)
        self._assert_equivalence(eval_annot, eval_qopt, annot_transpiled, qopt_transpiled)

    def _run_comparison_standard(  # pylint: disable=too-many-positional-arguments
        self, hamiltonian, cost_layer, num_qaoa_layers, backend, initial_layout, optimized=False
    ):
        """Run both transpilation pipelines and compare expectation values
        and number of final operations"""

        param_values = [5.11350346] * num_qaoa_layers + [5.52673212] * num_qaoa_layers
        standard_transpiled, annot_transpiled = self._run_standard_and_annot(
            cost_layer, hamiltonian, num_qaoa_layers, backend, initial_layout, optimized
        )
        eval_standard = self._estimate(standard_transpiled, hamiltonian, param_values)
        eval_annot = self._estimate(annot_transpiled, hamiltonian, param_values)

        self._assert_equivalence(eval_annot, eval_standard, annot_transpiled, standard_transpiled)

    def _run_all_cases(self, problem_fn, optimized=False):
        """Iterate over given test cases and run"""
        for layers, nodes, edges, elist in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = problem_fn(n=nodes, elist=elist)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self._run_comparison_qopt(
                    hamiltonian, cost_layer, layers, backend, initial_layout, optimized
                )

    def test_barabasi_albert_basic(self):
        """Run comparison with barabasi albert graph and no further optimizations."""
        for layers, nodes, edges, _ in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = get_problem_barabasi(n=nodes, m=edges)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self._run_comparison_qopt(hamiltonian, cost_layer, layers, backend, initial_layout)

    def test_barabasi_albert_optimized(self):
        """Run comparison with barabasi albert graph and an additional synthesis/cancellation step."""
        for layers, nodes, edges, _ in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = get_problem_barabasi(n=nodes, m=edges)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self._run_comparison_qopt(
                    hamiltonian, cost_layer, layers, backend, initial_layout, optimized=True
                )

    def test_maxcut_basic(self):
        """Run comparison with maxcut graph and no further optimizations."""
        self._run_all_cases(get_problem_maxcut)

    def test_maxcut_optimized(self):
        """Run comparison with maxcut graph and an additional synthesis/cancellation step."""
        self._run_all_cases(get_problem_maxcut, optimized=True)

    def test_standard(self):
        """Run comparison against standard pipeline."""
        hamiltonian = SparsePauliOp.from_list([("IZZ", 1), ("ZIZ", 2), ("ZZI", 3)])
        cost_layer = get_cost_layer(hamiltonian)
        backend, initial_layout = backend_and_layout_a(cost_layer)

        self._run_comparison_standard(
            hamiltonian, cost_layer, 1, backend, initial_layout, optimized=False
        )


if __name__ == "__main__":
    unittest.main()
