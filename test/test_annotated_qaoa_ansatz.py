"""Unit tests for the annotated qaoa ansatz function."""

import unittest
from networkx import barabasi_albert_graph

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qopt_best_practices.utils import build_max_cut_paulis
from qopt_best_practices.circuit_library import annotated_qaoa_ansatz


class TestAnnotatedQAOAAnsatz(unittest.TestCase):
    """Testthe annotated qaoa ansatz function."""

    def setUp(self):
        graph = barabasi_albert_graph(n=4, m=2, seed=42)
        paulis = build_max_cut_paulis(graph)
        self.hamiltonian = SparsePauliOp.from_list(paulis)

    def test_returns_quantum_circuit(self):
        """Test that the function returns a quantum circuit."""
        circuit = annotated_qaoa_ansatz(self.hamiltonian, reps=1)
        self.assertIsInstance(circuit, QuantumCircuit)

    def test_correct_number_of_qubits(self):
        """Test that the number of qubits matches the Hamiltonian."""
        circuit = annotated_qaoa_ansatz(self.hamiltonian, reps=1)
        self.assertEqual(circuit.num_qubits, self.hamiltonian.num_qubits)

    def test_box_structure(self):
        """Test that a 1 layer qaoa contains init_state, cost_layer and mixer boxes."""
        circuit = annotated_qaoa_ansatz(self.hamiltonian, reps=1)
        gate_names = [instr.operation.name for instr in circuit.data]
        self.assertEqual(gate_names, ["box", "box", "box"])
        for i, instr in enumerate(circuit.data):
            self.assertEqual(instr.operation.name, "box")
            if i == 0:
                self.assertEqual(instr.operation.annotations[0].namespace, "qaoa.init_state")
            elif i == 1:
                self.assertEqual(instr.operation.annotations[0].namespace, "qaoa.cost_layer")
            else:
                self.assertEqual(instr.operation.annotations[0].namespace, "qaoa.mixer")

    def test_zero_layers(self):
        """Test that 0 layers equals initial state box."""
        circuit = annotated_qaoa_ansatz(self.hamiltonian, reps=0)
        self.assertEqual(len(circuit.data), 1)

    def test_multiple_layers(self):
        """Test that 3 layers equals 3*2+1=7 boxes."""
        circuit = annotated_qaoa_ansatz(self.hamiltonian, reps=3)
        self.assertGreaterEqual(len(circuit.data), 7)
        for i, instr in enumerate(circuit.data):
            self.assertEqual(instr.operation.name, "box")
            if i < 3:
                self.assertEqual(instr.operation.annotations[0].payload, "1")
            elif i < 5:
                self.assertEqual(instr.operation.annotations[0].payload, "2")
            else:
                self.assertEqual(instr.operation.annotations[0].payload, "3")


if __name__ == "__main__":
    unittest.main()
