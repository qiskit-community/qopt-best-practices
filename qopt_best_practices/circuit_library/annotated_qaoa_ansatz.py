"""Annotated QAOA Ansatz builder"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Sequence
import itertools
import numpy as np

from qiskit.circuit import QuantumCircuit, annotation
from qiskit.circuit.parametervector import ParameterVector
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

if typing.TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis

from .pauli_evolution import PauliEvolutionGate  # pylint: disable=wrong-import-position


class CostLayerAnnotation(annotation.Annotation):
    """Annotation for cost layer boxes. The payload indicates the QAOA layer id,
    which is expected to start counting at 1! (not 0)"""

    def __init__(self, layer_id: int):
        self.namespace = "qaoa.cost_layer"
        self.payload = str(layer_id)


class MixerAnnotation(annotation.Annotation):
    """Annotation for mixer boxes. The payload indicates the QAOA layer id,
    which is expected to start counting at 1! (not 0)"""

    def __init__(self, layer_id: int):
        self.namespace = "qaoa.mixer"
        self.payload = str(layer_id)


class InitStateAnnotation(annotation.Annotation):
    """Annotation for initial state boxes. This annotation automatically sets
    the QAOA layer id as there should only be one initial state per QAOA circuit."""

    def __init__(self):
        self.namespace = "qaoa.init_state"
        self.payload = str(1)


def annotated_evolved_operator_ansatz(  # pylint: disable=too-many-positional-arguments
    operators: BaseOperator | Sequence[BaseOperator],
    reps: int = 1,
    evolution: EvolutionSynthesis | None = None,
    insert_barriers: bool = False,
    name: str = "EvolvedOps",
    parameter_prefix: str | Sequence[str] = "t",
    remove_identities: bool = True,
    flatten: bool | None = None,
    annotations: Sequence[annotation.Annotation] = None,
) -> QuantumCircuit:
    r"""Construct an ansatz out of operator evolutions with a series of annotations

    For a set of operators :math:`[O_1, ..., O_J]` and :math:`R` repetitions (``reps``), this circuit
    is defined as

    .. math::

        \prod_{r=1}^{R} \left( \prod_{j=J}^1 e^{-i\theta_{j, r} O_j} \right)

    where the exponentials :math:`exp(-i\theta O_j)` are expanded using the product formula
    specified by ``evolution``.

    Examples:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.circuit.library import evolved_operator_ansatz
        from qiskit.quantum_info import Pauli

        ops = [Pauli("ZZI"), Pauli("IZZ"), Pauli("IXI")]
        ansatz = evolved_operator_ansatz(ops, reps=3, insert_barriers=True)
        ansatz.draw("mpl")

    Args:
        operators: The operators to evolve. Can be a single operator or a sequence thereof.
        reps: The number of times to repeat the evolved operators.
        evolution: A specification of which evolution synthesis to use for the
            :class:`.PauliEvolutionGate`. Defaults to first order Trotterization. Note, that
            operators of type :class:`.Operator` are evolved using the :class:`.HamiltonianGate`,
            as there are no Hamiltonian terms to expand in Trotterization.
        insert_barriers: Whether to insert barriers in between each evolution.
        name: The name of the circuit.
        parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
            will be used for each parameters. Can also be a list to specify a prefix per
            operator.
        remove_identities: If ``True``, ignore identity operators (note that we do not check
            :class:`.Operator` inputs). This will also remove parameters associated with identities.
        flatten: If ``True``, a flat circuit is returned instead of nesting it inside multiple
            layers of gate objects. Setting this to ``False`` is significantly less performant,
            especially for parameter binding, but can be desirable for a cleaner visualization.
        annotations: A series of box annotations. For qaoa circuits, the expected order of annotations is
            alternating cost and mixer on a per-layer basis, with a total of
            ``reps * len(operators)`` elements:

            ```
                annotations = [CostLayerAnnotation(layer_id=1),
                                MixerAnnotation(layer_id=1),
                                CostLayerAnnotation(layer_id=2),
                                MixerAnnotation(layer_id=2),
                                ...]
            ```
    """

    if reps < 0:
        raise ValueError("reps must be a non-negative integer.")

    if isinstance(operators, BaseOperator):
        operators = [operators]
    elif len(operators) == 0:
        return QuantumCircuit()

    num_operators = len(operators)
    if not isinstance(parameter_prefix, str):
        if num_operators != len(parameter_prefix):
            raise ValueError(
                f"Mismatching number of operators ({len(operators)}) and parameter_prefix "
                f"({len(parameter_prefix)})."
            )

    num_qubits = operators[0].num_qubits
    if remove_identities:
        operators, parameter_prefix = _remove_identities(operators, parameter_prefix)

    if any(op.num_qubits != num_qubits for op in operators):
        raise ValueError("Inconsistent numbers of qubits in the operators.")

    # get the total number of parameters
    if isinstance(parameter_prefix, str):
        parameters = ParameterVector(parameter_prefix, reps * num_operators)
        param_iter = iter(parameters)
    else:
        # this creates the parameter vectors per operator, e.g.
        #    [[a0, a1, a2, ...], [b0, b1, b2, ...], [c0, c1, c2, ...]]
        # and turns them into an iterator
        #    a0 -> b0 -> c0 -> a1 -> b1 -> c1 -> a2 -> ...
        per_operator = [ParameterVector(prefix, reps).params for prefix in parameter_prefix]
        param_iter = itertools.chain.from_iterable(zip(*per_operator))

    # slower, Python-path
    if evolution is None:
        from qiskit.synthesis.evolution import LieTrotter

        evolution = LieTrotter(insert_barriers=insert_barriers)

    circuit = QuantumCircuit(num_qubits, name=name)

    # pylint: disable=cyclic-import
    from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate

    annot_index = 0
    for rep in range(reps):
        for i, op in enumerate(operators):
            if isinstance(op, Operator):
                gate = HamiltonianGate(op, next(param_iter))
                if flatten:
                    warnings.warn(
                        "Cannot flatten the evolution of an Operator, flatten is set to "
                        "False for this operator."
                    )
                flatten_operator = False

            elif isinstance(op, BaseOperator):
                gate = PauliEvolutionGate(op, next(param_iter), synthesis=evolution)
                flatten_operator = flatten is True or flatten is None
            else:
                raise ValueError(f"Unsupported operator type: {type(op)}")

            if flatten_operator:
                if annotations:
                    with circuit.box(annotations=(annotations[annot_index],)):
                        circuit.compose(gate.definition, inplace=True)
                    annot_index += 1
                else:
                    circuit.compose(gate.definition, inplace=True)

            else:
                if annotations:
                    with circuit.box(annotations=(annotations[annot_index],)):
                        circuit.append(gate, circuit.qubits)
                    annot_index += 1
                else:
                    circuit.append(gate, circuit.qubits)

            if insert_barriers and (rep < reps - 1 or i < num_operators - 1):
                circuit.barrier()

    return circuit


def _is_pauli_identity(operator):
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]  # check if the single Pauli is identity below
        else:
            return False
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False


def _remove_identities(operators, prefixes):
    identity_ops = {index for index, op in enumerate(operators) if _is_pauli_identity(op)}

    if len(identity_ops) == 0:
        return operators, prefixes

    cleaned_ops = [op for i, op in enumerate(operators) if i not in identity_ops]
    cleaned_prefix = [prefix for i, prefix in enumerate(prefixes) if i not in identity_ops]

    return cleaned_ops, cleaned_prefix


def annotated_qaoa_ansatz(  # pylint: disable=too-many-positional-arguments
    cost_operator: BaseOperator,
    reps: int = 1,
    initial_state: QuantumCircuit | None = None,
    mixer_operator: BaseOperator | None = None,
    insert_barriers: bool = False,
    name: str = "QAOA",
    flatten: bool = True,
) -> QuantumCircuit:
    r"""A generalized annotated QAOA quantum circuit with a support of custom initial states and mixers.
        The annotations indicate the layer number, and whether the layer is a cost layer or a mixer.

    Examples:

        To define the QAOA ansatz we require a cost Hamiltonian, encoding the classical
        optimization problem:

        .. plot::
            :alt: Circuit diagram output by the previous code.
            :include-source:

            from qiskit.quantum_info import SparsePauliOp
            from qiskit.circuit.library import qaoa_ansatz

            cost_operator = SparsePauliOp(["ZZII", "IIZZ", "ZIIZ"])
            ansatz = qaoa_ansatz(cost_operator, reps=3, insert_barriers=True)
            ansatz.draw("mpl")

    Args:
        cost_operator: The operator representing the cost of the optimization problem, denoted as
            :math:`U(C, \gamma)` in [1].
        reps: The integer determining the depth of the circuit, called :math:`p` in [1].
        initial_state: An optional initial state to use, which defaults to a layer of
            Hadamard gates preparing the :math:`|+\rangle^{\otimes n}` state.
            If a custom mixer is chosen, this circuit should be set to prepare its ground state,
            to appropriately fulfill the annealing conditions.
        mixer_operator: An optional custom mixer, which defaults to global Pauli-:math:`X`
            rotations. This is denoted as :math:`U(B, \beta)` in [1]. If this is set,
            the ``initial_state`` might also require modification.
        insert_barriers: Whether to insert barriers in-between the cost and mixer operators.
        name: The name of the circuit.
        flatten: If ``True``, a flat circuit is returned instead of nesting it inside multiple
            layers of gate objects. Setting this to ``False`` is significantly less performant,
            especially for parameter binding, but can be desirable for a cleaner visualization.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """
    num_qubits = cost_operator.num_qubits

    circuit = QuantumCircuit(num_qubits)
    with circuit.box(annotations=(InitStateAnnotation(),)):
        if initial_state is None:
            initial_state = QuantumCircuit(num_qubits)
            initial_state.h(range(num_qubits))
        circuit.compose(initial_state, inplace=True)

    if mixer_operator is None:
        mixer_operator = SparsePauliOp.from_sparse_list(
            [("X", [i], 1) for i in range(num_qubits)], num_qubits
        )

    parameter_prefix = ["γ", "β"]

    annotations = []
    for layer_id in range(1, reps + 1):
        annotations += [CostLayerAnnotation(layer_id), MixerAnnotation(layer_id)]
    out_circuit = circuit.compose(
        annotated_evolved_operator_ansatz(
            [cost_operator, mixer_operator],
            reps=reps,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
            flatten=flatten,
            annotations=annotations,
        ),
        copy=False,
    )

    for inst in out_circuit:
        if inst.operation.name == "box":
            if inst.operation.num_qubits != out_circuit.num_qubits:
                raise NotImplementedError("This constructor does not support incomplete graphs.")
    return out_circuit
