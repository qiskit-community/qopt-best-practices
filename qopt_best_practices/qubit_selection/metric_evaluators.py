"""Subset finders. Currently contains reference implementation
to evaluate 2-qubit gate fidelity."""

from __future__ import annotations
from qiskit.providers import Backend
import rustworkx as rx

TWO_Q_GATES = ["cx", "ecr", "cz"]


def evaluate_fidelity(path: list[int], backend: Backend, edges: rx.EdgeList) -> float:
    """Evaluates fidelity on a given list of qubits based on the two-qubit gate error
    for a specific backend.

    Returns:
       Path fidelity.
    """

    two_qubit_fidelity = {}

    if backend.version == 2:
        target = backend.target
        try:
            gate_name = list(set(TWO_Q_GATES).intersection(backend.operation_names))[0]
        except IndexError as exc:
            raise ValueError("Could not identify two-qubit gate") from exc

        for edge in edges:
            try:
                cx_error = target[gate_name][edge].error
            except:  # pylint: disable=bare-except
                cx_error = target[gate_name][edge[::-1]].error

            two_qubit_fidelity[tuple(edge)] = 1 - cx_error
    else:
        props = backend.properties()
        try:
            gate_name = list(set(TWO_Q_GATES).intersection(backend.configuration().basis_gates))[0]
        except IndexError as exc:
            raise ValueError("Could not identify two-qubit gate") from exc

        for edge in edges:
            try:
                cx_error = props.gate_error(gate_name, edge)
            except:  # pylint: disable=bare-except
                cx_error = props.gate_error(gate_name, edge[::-1])

            two_qubit_fidelity[tuple(edge)] = 1 - cx_error

    if not path or len(path) == 1:
        return 0.0

    fidelity = 1.0
    for idx in range(len(path) - 1):
        fidelity *= two_qubit_fidelity[(path[idx], path[idx + 1])]
    return fidelity
