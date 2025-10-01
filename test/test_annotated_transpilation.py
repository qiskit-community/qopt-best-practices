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
from qiskit.circuit.library import CXGate
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
    graph = barabasi_albert_graph(n=n, m=m, seed=42)
    local_correlators = build_max_cut_paulis(graph)
    cost_operator = SparsePauliOp.from_list(local_correlators)
    cost_layer = get_cost_layer(cost_operator)
    return cost_operator, cost_layer, graph


def get_problem_maxcut(n=4, elist=None):
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

    def qopt_vs_annot_basic(
        self,
        cost_layer,
        hamiltonian,
        num_qaoa_layers,
        backend,
        initial_layout,
        estimator,
        param_values,
    ):
        swap_strategy = SwapStrategy.from_line(list(range(cost_layer.num_qubits)))
        edge_coloring = {(i, i + 1): (i + 1) % 2 for i in range(cost_layer.num_qubits)}

        # QOpt pipeline
        pre_init_qopt = PassManager(
            [
                PrepareCostLayer(),
                Commuting2qGateRouter(swap_strategy, edge_coloring),
                SwapToFinalMapping(),
                QAOAConstructionPass(num_layers=num_qaoa_layers),
            ]
        )
        qopt_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        qopt_pm.pre_init = pre_init_qopt
        qopt_transpiled = qopt_pm.run(cost_layer)
        isa_hamiltonian_qopt = hamiltonian.apply_layout(qopt_transpiled.layout)
        qopt_transpiled.remove_final_measurements()
        eval_qopt = (
            estimator.run([(qopt_transpiled, isa_hamiltonian_qopt, param_values)])
            .result()[0]
            .data.values()
        )

        # Annotated pipeline
        annotated_ansatz = annotated_qaoa_ansatz(hamiltonian, reps=num_qaoa_layers)
        pre_init_annot = PassManager(
            [
                AnnotatedPrepareCostLayer(),
                AnnotatedCommuting2qGateRouter(swap_strategy, edge_coloring),
                AnnotatedSwapToFinalMapping(),
                UnrollBoxes(),
            ]
        )
        annot_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        annot_pm.pre_init = pre_init_annot
        annot_transpiled = annot_pm.run(annotated_ansatz)
        isa_hamiltonian_annot = hamiltonian.apply_layout(annot_transpiled.layout)
        annot_transpiled.remove_final_measurements()
        eval_annot = (
            estimator.run([(annot_transpiled, isa_hamiltonian_annot, param_values)])
            .result()[0]
            .data.values()
        )
        print("EVALS:", list(eval_annot), list(eval_qopt))
        self.assertEqual(list(eval_annot), list(eval_qopt))
        for key in annot_transpiled.count_ops():
            self.assertEqual(annot_transpiled.count_ops()[key], qopt_transpiled.count_ops()[key])

    def run_basic_test(self, hamiltonian, cost_layer, num_qaoa_layers, backend, initial_layout):
        optimal_gamma = [5.11350346] * num_qaoa_layers
        optimal_beta = [5.52673212] * num_qaoa_layers
        param_values = optimal_gamma + optimal_beta
        self.qopt_vs_annot_basic(
            cost_layer,
            hamiltonian,
            num_qaoa_layers,
            backend,
            initial_layout,
            self.estimator,
            param_values,
        )

    def qopt_vs_annot_optimized(
        self,
        cost_layer,
        hamiltonian,
        num_qaoa_layers,
        backend,
        initial_layout,
        estimator,
        param_values,
    ):
        swap_strategy = SwapStrategy.from_line(list(range(cost_layer.num_qubits)))
        edge_coloring = {(i, i + 1): (i + 1) % 2 for i in range(cost_layer.num_qubits)}

        # QOpt pipeline
        pre_init_qopt = PassManager(
            [
                PrepareCostLayer(),
                Commuting2qGateRouter(swap_strategy, edge_coloring),
                SwapToFinalMapping(),
                HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
                InverseCancellation(gates_to_cancel=[CXGate()]),
                QAOAConstructionPass(num_layers=num_qaoa_layers),
            ]
        )
        qopt_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        qopt_pm.pre_init = pre_init_qopt
        qopt_transpiled = qopt_pm.run(cost_layer)
        isa_hamiltonian_qopt = hamiltonian.apply_layout(qopt_transpiled.layout)
        qopt_transpiled.remove_final_measurements()
        eval_qopt = (
            estimator.run([(qopt_transpiled, isa_hamiltonian_qopt, param_values)])
            .result()[0]
            .data.values()
        )

        # Annotated pipeline
        annotated_ansatz = annotated_qaoa_ansatz(hamiltonian, reps=num_qaoa_layers)
        pre_init_annot = PassManager(
            [
                AnnotatedPrepareCostLayer(),
                AnnotatedCommuting2qGateRouter(swap_strategy, edge_coloring),
                AnnotatedSwapToFinalMapping(),
                SynthesizeAndSimplifyCostLayer(basis_gates=["x", "cx", "sx", "rz", "id"]),
                UnrollBoxes(),
            ]
        )
        annot_pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, initial_layout=initial_layout
        )
        annot_pm.pre_init = pre_init_annot
        annot_transpiled = annot_pm.run(annotated_ansatz)
        isa_hamiltonian_annot = hamiltonian.apply_layout(annot_transpiled.layout)
        annot_transpiled.remove_final_measurements()
        eval_annot = (
            estimator.run([(annot_transpiled, isa_hamiltonian_annot, param_values)])
            .result()[0]
            .data.values()
        )
        print("EVALS:", list(eval_annot), list(eval_qopt))
        self.assertEqual(list(eval_annot), list(eval_qopt))
        for key in annot_transpiled.count_ops():
            self.assertEqual(annot_transpiled.count_ops()[key], qopt_transpiled.count_ops()[key])

    def run_optimized_test(self, hamiltonian, cost_layer, num_qaoa_layers, backend, initial_layout):
        optimal_gamma = [5.11350346] * num_qaoa_layers
        optimal_beta = [5.52673212] * num_qaoa_layers
        param_values = optimal_gamma + optimal_beta
        self.qopt_vs_annot_optimized(
            cost_layer,
            hamiltonian,
            num_qaoa_layers,
            backend,
            initial_layout,
            self.estimator,
            param_values,
        )

    def test_barabasi_albert_basic(self):
        for layers, nodes, edges, _ in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = get_problem_barabasi(n=nodes, m=edges)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self.run_basic_test(hamiltonian, cost_layer, layers, backend, initial_layout)

    def test_maxcut_basic(self):
        for layers, nodes, edges, elist in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = get_problem_maxcut(n=nodes, elist=elist)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self.run_basic_test(hamiltonian, cost_layer, layers, backend, initial_layout)

    def test_barabasi_albert_optimized(self):
        for layers, nodes, edges, _ in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = get_problem_barabasi(n=nodes, m=edges)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self.run_optimized_test(hamiltonian, cost_layer, layers, backend, initial_layout)

    def test_maxcut_optimized(self):
        for layers, nodes, edges, elist in self.test_cases:
            with self.subTest(layers=layers, nodes=nodes, edges=edges):
                hamiltonian, cost_layer, _ = get_problem_maxcut(n=nodes, elist=elist)
                backend, initial_layout = backend_and_layout_a(cost_layer)
                self.run_optimized_test(hamiltonian, cost_layer, layers, backend, initial_layout)


if __name__ == "__main__":
    unittest.main()
