"""PauliEvolutionGate subclass that supports ParameterExpression operator coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import ParameterValueType
from qiskit.quantum_info import Pauli, SparsePauliOp, SparseObservable
from qiskit.circuit.library.pauli_evolution import (
    PauliEvolutionGate as _QiskitPauliEvolutionGate,
    _get_default_label,
)

if TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis


def _to_sparse_op(
    operator: Pauli | SparsePauliOp | SparseObservable,
) -> SparsePauliOp | SparseObservable:
    """Like Qiskit's _to_sparse_op but allows ParameterExpression coefficients."""
    if isinstance(operator, Pauli):
        sparse = SparsePauliOp(operator)
    elif isinstance(operator, (SparseObservable, SparsePauliOp)):
        sparse = operator
    else:
        raise ValueError(f"Unsupported operator type for evolution: {type(operator)}.")

    if any(np.iscomplex(c) for c in sparse.coeffs if not isinstance(c, ParameterExpression)):
        raise ValueError("Operator contains complex coefficients, which are not supported.")

    return sparse


class PauliEvolutionGate(_QiskitPauliEvolutionGate):
    """PauliEvolutionGate extended to support ParameterExpression operator coefficients.

    Upstream Qiskit rejects ParameterExpression values as Hamiltonian coefficients.
    This subclass removes that restriction to enable multi-objective optimization (MOO)
    workflows where the cost Hamiltonian weights are themselves circuit parameters.

    All other behaviour is inherited from :class:`qiskit.circuit.library.PauliEvolutionGate`.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        operator: (
            Pauli
            | SparsePauliOp
            | SparseObservable
            | list[Pauli | SparsePauliOp | SparseObservable]
        ),
        time: ParameterValueType = 1.0,
        label: str | None = None,
        synthesis: EvolutionSynthesis | None = None,
    ) -> None:
        if isinstance(operator, list):
            operator = [_to_sparse_op(op) for op in operator]
        else:
            operator = _to_sparse_op(operator)

        if label is None:
            label = _get_default_label(operator)

        if isinstance(operator, list):
            if not operator:
                raise ValueError("The argument 'operator' cannot be an empty list.")
            num_qubits = operator[0].num_qubits
            for op in operator[1:]:
                if op.num_qubits != num_qubits:
                    raise ValueError(
                        "When represented as a list of operators, all of these operators "
                        "must have the same number of qubits."
                    )
        else:
            num_qubits = operator.num_qubits

        # Call Gate.__init__ directly to bypass _QiskitPauliEvolutionGate.__init__, which
        # would invoke its own _to_sparse_op and reject ParameterExpression coefficients.
        Gate.__init__(  # pylint: disable=non-parent-init-called
            self, name="PauliEvolution", num_qubits=num_qubits, params=[time], label=label
        )
        self.operator = operator

        if synthesis is None:
            from qiskit.synthesis.evolution import LieTrotter  # pylint: disable=cyclic-import

            synthesis = LieTrotter()

        self.synthesis = synthesis
