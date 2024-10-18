"""The the cancellation of unneeded SWAP gates."""


from unittest import TestCase

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Layout

from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


class TestSwapCancellation(TestCase):
    """Test the swap cancellation pass."""

    def test_simple(self):
        """Simple test."""

        qc = QuantumCircuit(4, 4)
        qc.swap(3, 2)
        qc.rzz(1.234, 1, 2)
        qc.swap(1, 2)
        
        qreg = next(iter(qc.qregs))

        swap_pass = SwapToFinalMapping()
        layout = Layout(
            {
                0: qreg[0],
                1: qreg[2],
                2: qreg[3],
                3: qreg[1],
            }
        )
        swap_pass.property_set["virtual_permutation_layout"] = layout

        dag = circuit_to_dag(qc)
        qc2 = dag_to_circuit(swap_pass.run(dag))

        new_layout = swap_pass.property_set["virtual_permutation_layout"]

        self.assertTrue(new_layout[0] == qreg[0])
        self.assertTrue(new_layout[1] == qreg[1])
        self.assertTrue(new_layout[2] == qreg[3])
        self.assertTrue(new_layout[3] == qreg[2])
