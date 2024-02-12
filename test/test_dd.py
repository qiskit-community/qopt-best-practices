"""Tests dynamical decoupling."""

from unittest import TestCase

from qiskit import QuantumCircuit
from qiskit.transpiler import InstructionDurations
from qiskit.providers.fake_provider import FakeSherbrooke

from qopt_best_practices.transpilation import dd_pass_manager


class TestDynamicalDecoupling(TestCase):
    """Test the dynamical decoupling sequences."""

    def setUp(self):
        """Initialize variables we need."""
        self.durations = InstructionDurations([("cx", None, 800), ("x", None, 50)])

    def test_xx(self):
        """Test the standard passes."""

        dd_pass = dd_pass_manager(FakeSherbrooke(), "XX", durations=self.durations)

        qc = QuantumCircuit(2)
        qc.x([0, 1])
        qc.delay(1000, [0, 1])

        qc = dd_pass.run(qc)