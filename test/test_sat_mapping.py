"""Tests for SAT Mapping Utils"""

import json
import os
from unittest import TestCase

import networkx as nx
from qiskit.circuit import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.sat_mapping import SATMapper
from qopt_best_practices.utils import build_max_cut_graph, build_max_cut_paulis


class TestSwapStrategies(TestCase):
    """Unit test for SWAP strategies functionality."""

    def setUp(self):
        super().setUp()

        # load data
        graph_file = os.path.join(os.path.dirname(__file__), "data/graph_2layers_0seed.json")

        with open(graph_file, "r") as file:
            data = json.load(file)

        self.original_graph = nx.from_edgelist(data["Original graph"])
        self.original_paulis = build_max_cut_paulis(self.original_graph)

        self.mapped_paulis = [tuple(pauli) for pauli in data["paulis"]]
        self.mapped_graph = build_max_cut_graph(self.mapped_paulis)

        self.sat_mapping = {int(key): value for key, value in data["SAT mapping"].items()}
        self.min_k = data["min swap layers"]
        self.swap_strategy = SwapStrategy.from_line(list(range(len(self.original_graph.nodes))))
        self.basic_graphs = [nx.path_graph(5), nx.cycle_graph(7)]

    def test_find_initial_mappings(self):
        """Test find_initial_mappings"""

        mapper = SATMapper()

        results = mapper.find_initial_mappings(self.original_graph, self.swap_strategy)
        min_k = min((k for k, v in results.items() if v.satisfiable))
        # edge_map = dict(results[min_k].mapping)

        # edge maps are not equal, but same min_k
        self.assertEqual(min_k, self.min_k)

        # Find better test
        # self.assertEqual(edge_map, self.sat_mapping)

    def test_remap_graph_with_sat(self):
        """Test remap_graph_with_sat"""

        mapper = SATMapper()

        remapped_g, _, _ = mapper.remap_graph_with_sat(
            graph=self.original_graph, swap_strategy=self.swap_strategy
        )

        self.assertTrue(nx.is_isomorphic(remapped_g, self.mapped_graph))

    def test_deficient_strategy(self):
        """Test that the SAT mapper works when the SWAP strategy is deficient.

        Note: a deficient strategy does not result in full connectivity but
        may still be useful.
        """
        cmap = CouplingMap([(idx, idx + 1) for idx in range(10)])

        # This swap strategy is deficient but can route the graph below.
        swaps = (
            ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)),
            ((1, 2), (3, 4), (5, 6), (7, 8)),
            ((2, 3), (4, 5), (6, 7), (8, 9)),
            (),
            (),
            (),
            (),
            (),
        )
        swap_strategy = SwapStrategy(cmap, swaps)
        graph = nx.random_regular_graph(3, 10, seed=2)

        mapper = SATMapper()

        _, permutation, min_layer = mapper.remap_graph_with_sat(graph, swap_strategy)

        # Spot check a few permutations.
        self.assertEqual(permutation[0], 9)
        self.assertEqual(permutation[8], 1)

        # Crucially, if the `connectivity_matrix` in `find_initial_mappings` we get a wrong result.
        self.assertEqual(min_layer, 3)

    def test_full_connectivity(self):
        """Test that the SAT mapper works when the SWAP strategy has full connectivity."""
        graph = nx.random_regular_graph(3, 6, seed=1)
        swap_strategy = SwapStrategy.from_line(list(range(6)))
        sat_mapper = SATMapper()
        _, _, min_sat_layers = sat_mapper.remap_graph_with_sat(
            graph=graph,
            swap_strategy=swap_strategy,
        )
        self.assertEqual(min_sat_layers, 4)

    def test_unable_to_remap(self):
        """Test that the SAT mapper works when the SWAP strategy is unable to remap."""
        graph = nx.random_regular_graph(3, 6, seed=1)
        cmap = CouplingMap([(idx, idx + 1) for idx in range(5)])
        swap_strategy = SwapStrategy(cmap, [])
        sat_mapper = SATMapper()
        remapped_g, edge_map, min_sat_layers = sat_mapper.remap_graph_with_sat(
            graph=graph,
            swap_strategy=swap_strategy,
        )
        self.assertIsNone(remapped_g)
        self.assertIsNone(edge_map)
        self.assertIsNone(min_sat_layers)

    def test_parametric_hamiltonian(self):
        """Test that SATMapper preserves parametric Hamiltonians correctly.

        Verifies that when a parametric cost Hamiltonian (with Parameter objects as
        coefficients) is passed to remap_graph_with_sat, the parameters are preserved
        in the remapped operator with the correct qubit mapping. This is critical for
        multi-objective workflows where parameters are bound later during optimization.
        """
        # Create a simple parametric Hamiltonian with Parameter weights
        # H = c_0 * ZZ_01 + c_1 * ZZ_12 + c_2 * ZZ_23
        c_0 = Parameter("c_0")
        c_1 = Parameter("c_1")
        c_2 = Parameter("c_2")

        # Create parametric SparsePauliOp using direct constructor (not from_list) to ensure parameters are preserved
        pauli_strings = ["ZZII", "IZZI", "IIZZ"]
        coeffs = [c_0, c_1, c_2]
        parametric_hamiltonian = SparsePauliOp(pauli_strings, coeffs)

        # Create a swap strategy for 4 qubits
        swap_strategy = SwapStrategy.from_line([0, 1, 2, 3])

        mapper = SATMapper()

        # SATMapper should handle parametric Hamiltonians and preserve parameters
        remapped_op, edge_map, min_layers = mapper.remap_graph_with_sat(
            parametric_hamiltonian, swap_strategy
        )

        # Verify results - the mapping should succeed
        self.assertIsNotNone(remapped_op)
        self.assertIsNotNone(edge_map)
        self.assertIsNotNone(min_layers)
        self.assertIsInstance(remapped_op, SparsePauliOp)
        self.assertIsInstance(edge_map, dict)
        self.assertIsInstance(min_layers, int)

        # Verify the remapped operator still has parametric coefficients
        self.assertTrue(any(isinstance(coeff, ParameterExpression) for coeff in remapped_op.coeffs))

        # Verify all original parameters are present in remapped operator
        original_params = set(parametric_hamiltonian.parameters)
        remapped_params = set(remapped_op.parameters)
        self.assertEqual(original_params, remapped_params)

    def test_non_parametric_operator(self):
        """Test SATMapper with non-parametric SparsePauliOp.

        Verifies that the new parametric support doesn't break existing
        functionality for numeric (non-parametric) operators.
        """
        # Create non-parametric Hamiltonian with numeric weights
        pauli_strings = ["ZZII", "IZZI", "IIZZ"]
        coeffs = [1.5, 2.0, 0.5]  # Different weights to verify preservation
        numeric_hamiltonian = SparsePauliOp(pauli_strings, coeffs)

        # Create swap strategy
        swap_strategy = SwapStrategy.from_line([0, 1, 2, 3])

        mapper = SATMapper()

        # Apply SAT mapping
        remapped_op, edge_map, min_layers = mapper.remap_graph_with_sat(
            numeric_hamiltonian, swap_strategy
        )

        # Verify results
        self.assertIsNotNone(remapped_op)
        self.assertIsNotNone(edge_map)
        self.assertIsNotNone(min_layers)
        self.assertIsInstance(remapped_op, SparsePauliOp)
        self.assertIsInstance(edge_map, dict)
        self.assertIsInstance(min_layers, int)

        # Verify no parameters in output (all numeric)
        self.assertEqual(len(remapped_op.parameters), 0)

        # Verify coefficients are preserved (not all 1.0)
        coeffs_list = [abs(c) for c in remapped_op.coeffs]
        self.assertEqual({1.5, 2.0, 0.5}, set(coeffs_list))

    def test_parametric_hamiltonian_with_numeric_fallback(self):
        """Test SATMapper with parametric Hamiltonian using numeric values.

        This test verifies that if we bind parameters to numeric values first,
        the SAT mapping works correctly.
        """
        # Create parametric Hamiltonian
        c_0 = Parameter("c_0")
        c_1 = Parameter("c_1")
        c_2 = Parameter("c_2")

        pauli_strings = ["ZZII", "IZZI", "IIZZ"]
        coeffs = [c_0, c_1, c_2]
        parametric_hamiltonian = SparsePauliOp(pauli_strings, coeffs)

        # Bind parameters to numeric values
        param_dict = {c_0: 1.0, c_1: 1.0, c_2: 1.0}
        numeric_hamiltonian = parametric_hamiltonian.assign_parameters(param_dict)

        # Verify numeric values before mapping
        coeffs_before = [abs(c) for c in numeric_hamiltonian.coeffs]
        self.assertEqual({1.0}, set(coeffs_before))
        self.assertEqual(len(numeric_hamiltonian.parameters), 0)

        # Create swap strategy
        swap_strategy = SwapStrategy.from_line([0, 1, 2, 3])

        mapper = SATMapper()

        # This should work with numeric values
        remapped_op, edge_map, min_layers = mapper.remap_graph_with_sat(
            numeric_hamiltonian, swap_strategy
        )

        # Verify results
        self.assertIsNotNone(remapped_op)
        self.assertIsNotNone(edge_map)
        self.assertIsNotNone(min_layers)
        self.assertIsInstance(remapped_op, SparsePauliOp)
        self.assertIsInstance(edge_map, dict)
        coeffs_after = [abs(c) for c in remapped_op.coeffs]
        self.assertEqual({1.0}, set(coeffs_after))

    def test_remap_operator_large_qubits(self):
        """Test remap_operator with large number of qubits to verify to_label() doesn't truncate."""
        num_qubits = 50

        # Create operator with ZZ terms on qubits 0-1 and 48-49
        pauli_strings = ["ZZ" + "I" * 48, "I" * 48 + "ZZ"]
        coeffs = [1.0, 2.0]
        large_operator = SparsePauliOp(pauli_strings, coeffs)

        # Remap qubits
        qubit_map = {0: 10, 1: 11, 48: 20, 49: 21}
        remapped_op = SATMapper.remap_operator(large_operator, qubit_map)

        # Verify structure
        self.assertEqual(remapped_op.num_qubits, num_qubits)
        self.assertEqual(len(remapped_op), 2)

        # Verify coefficients preserved
        coeffs_list = [abs(c) for c in remapped_op.coeffs]
        self.assertEqual({1.0, 2.0}, set(coeffs_list))

        # Verify no truncation - each label should have exactly 2 Z's and correct length
        labels = [pauli.to_label() for pauli in remapped_op.paulis]
        for label in labels:
            self.assertEqual(label.count("Z"), 2)
            self.assertEqual(len(label), num_qubits)
        self.assertEqual(len(remapped_op.parameters), 0)
