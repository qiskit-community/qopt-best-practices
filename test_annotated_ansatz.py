from __future__ import annotations

from collections.abc import Sequence

import typing
import warnings
import itertools
import numpy as np

from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.synthesis.evolution.product_formula import real_or_fail
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit._accelerate.circuit_library import pauli_evolution

from qiskit.circuit.library.n_local import NLocal
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    # Commuting2qGateRouter,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
    UnrollCustomDefinitions,
    HighLevelSynthesis,
    InverseCancellation
)

from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.circuit.library import CXGate
from qiskit.transpiler import TransformationPass

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)



if typing.TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis
    
from qiskit.circuit import annotation, QuantumCircuit

class CostLayerAnnotation(annotation.Annotation):
    namespace = "qaoa.cost_layer"

class MixerAnnotation(annotation.Annotation):
    namespace = "qaoa.mixer"

class InitStateAnnotation(annotation.Annotation):
    namespace = "qaoa.init_state"

def annotated_evolved_operator_ansatz(
    operators: BaseOperator | Sequence[BaseOperator],
    reps: int = 1,
    evolution: EvolutionSynthesis | None = None,
    insert_barriers: bool = False,
    name: str = "EvolvedOps",
    parameter_prefix: str | Sequence[str] = "t",
    remove_identities: bool = True,
    flatten: bool | None = None,
    annotations: Sequence[annotation.Annotation]=None,
) -> QuantumCircuit:
    r"""Construct an ansatz out of operator evolutions.

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

    # # fast, Rust-path
    # if (
    #     flatten is not False  # captures None and True
    #     and evolution is None
    #     and all(isinstance(op, SparsePauliOp) for op in operators)
    # ):
    #     sparse_labels = [op.to_sparse_list() for op in operators]
    #     expanded_paulis = []
    #     for _ in range(reps):
    #         for term in sparse_labels:
    #             param = next(param_iter)
    #             expanded_paulis += [
    #                 (pauli, indices, 2 * real_or_fail(coeff) * param)
    #                 for pauli, indices, coeff in term
    #             ]

    #     data = pauli_evolution(num_qubits, expanded_paulis, insert_barriers, False)
    #     circuit = QuantumCircuit._from_circuit_data(data, legacy_qubits=True)
    #     circuit.name = name

    #     return circuit

    # slower, Python-path
    if evolution is None:
        from qiskit.synthesis.evolution import LieTrotter

        evolution = LieTrotter(insert_barriers=insert_barriers)

    circuit = QuantumCircuit(num_qubits, name=name)

    # pylint: disable=cyclic-import
    from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate

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

            print("AM I PRINTING")
            if flatten_operator:
                if annotations:
                    print("USE BOX")
                    with circuit.box(annotations=(annotations[i],)):
                        circuit.compose(gate.definition, inplace=True)
                else:
                    circuit.compose(gate.definition, inplace=True)

            else:
                if annotations:
                    print("USE BOX")
                    with circuit.box(annotations=(annotations[i],)):
                        circuit.append(gate, circuit.qubits)
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

def annotated_qaoa_ansatz(
    cost_operator: BaseOperator,
    reps: int = 1,
    initial_state: QuantumCircuit | None = None,
    mixer_operator: BaseOperator | None = None,
    insert_barriers: bool = False,
    name: str = "QAOA",
    flatten: bool = True,
) -> QuantumCircuit:
    r"""A generalized QAOA quantum circuit with a support of custom initial states and mixers.

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

    if initial_state is None:
        initial_state = QuantumCircuit(num_qubits)
        initial_state.h(range(num_qubits))

    if mixer_operator is None:
        mixer_operator = SparsePauliOp.from_sparse_list(
            [("X", [i], 1) for i in range(num_qubits)], num_qubits
        )

    parameter_prefix = ["γ", "β"]

    annotations = [CostLayerAnnotation(), MixerAnnotation()]
    return initial_state.compose(
        annotated_evolved_operator_ansatz(
            [cost_operator, mixer_operator],
            reps=reps,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
            flatten=flatten,
            annotations=annotations
        ),
        copy=False,
    )

from qiskit.converters import dag_to_circuit, circuit_to_dag
class PrepareCostLayer(TransformationPass):
    """Prepares the cost layer for the `Commuting2qGateRouter`.

    The cost layer may have single qubit gates, two qubit gates, and more.
    The `qaoa_ansatz` class from the Qiskit circuit library will add Rz
    rotations for first-order terms and Rzz rotations for second-order terms.
    The `Commuting2qGateRouter` expects an instance of `Commuting2qBlock` in
    the cost layer to route using swap networks. This pass will group all the
    two-qubit Rzz gates together into such an instruction and keep the Rz
    rotations separate.

    Note that high order terms (i.e. cubic and more) produce ladders of
    CX gates with a Rz rotation when using `qaoa_ansatz`. This pass currently
    does not support high order terms.
    """

    def run(self, dag):
        """Run the pass."""
        for node in dag.topological_op_nodes():
            if node.op.name == "box":
                if "cost_layer" in node.op.annotations[0].namespace:
                    box_dag = circuit_to_dag(node.op.params[0])
                    commuting_nodes = []
                    for box_node in box_dag.topological_op_nodes():
                        if box_node.op.name not in ["rz", "rzz"]:
                            raise ValueError(
                                f"{self.__class__.__name__} only supports rz and rzz nodes. "
                                f"Found {box_node.op.name} instead."
                            )

                        if box_node.op.name == "rzz":
                            commuting_nodes.append(box_node)

                    commuting_block = Commuting2qBlock(commuting_nodes)

                    wire_order = {
                        wire: idx for idx, wire in enumerate(box_dag.qubits) if wire not in box_dag.idle_wires()
                    }

                    box_dag.replace_block_with_op(commuting_nodes, commuting_block, wire_order)
                    node.op.params[0] = dag_to_circuit(box_dag)
                    dag.substitute_node_with_dag(node, box_dag)
                else:
                    box_dag = circuit_to_dag(node.op.params[0])
                    dag.substitute_node_with_dag(node, box_dag)



from collections import defaultdict

from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)


class Commuting2qGateRouter(TransformationPass):
    """A class to swap route one or more commuting gates to the coupling map.

    This pass routes blocks of commuting two-qubit gates encapsulated as
    :class:`.Commuting2qBlock` instructions. This pass will not apply to other instructions.
    The mapping to the coupling map is done using swap strategies, see :class:`.SwapStrategy`.
    The swap strategy should suit the problem and the coupling map. This transpiler pass
    should ideally be executed before the quantum circuit is enlarged with any idle ancilla
    qubits. Otherwise, we may swap qubits outside the portion of the chip we want to use.
    Therefore, the swap strategy and its associated coupling map do not represent physical
    qubits. Instead, they represent an intermediate mapping that corresponds to the physical
    qubits once the initial layout is applied. The example below shows how to map a four
    qubit :class:`.PauliEvolutionGate` to qubits 0, 1, 3, and 4 of the five qubit device with
    the coupling map

    .. code-block:: text

        0 -- 1 -- 2
             |
             3
             |
             4

    To do this we use a line swap strategy for qubits 0, 1, 3, and 4 defined it in terms
    of virtual qubits 0, 1, 2, and 3.

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler import Layout, CouplingMap, PassManager
        from qiskit.transpiler.passes import FullAncillaAllocation
        from qiskit.transpiler.passes import EnlargeWithAncilla
        from qiskit.transpiler.passes import ApplyLayout
        from qiskit.transpiler.passes import SetLayout

        from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
            SwapStrategy,
            FindCommutingPauliEvolutions,
            Commuting2qGateRouter,
        )

        # Define the circuit on virtual qubits
        op = SparsePauliOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        # Define the swap strategy on qubits before the initial_layout is applied.
        swap_strat = SwapStrategy.from_line([0, 1, 2, 3])

        # Chose qubits 0, 1, 3, and 4 from the backend coupling map shown above.
        backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4)])
        initial_layout = Layout.from_intlist([0, 1, 3, 4], *circ.qregs)

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
                SetLayout(initial_layout),
                FullAncillaAllocation(backend_cmap),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )

        # Insert swap gates, map to initial_layout and finally enlarge with ancilla.
        pm_pre.run(circ).draw("mpl")

    This pass manager relies on the ``current_layout`` which corresponds to the qubit layout as
    swap gates are applied. The pass will traverse all nodes in the dag. If a node should be
    routed using a swap strategy then it will be decomposed into sub-instructions with swap
    layers in between and the ``current_layout`` will be modified. Nodes that should not be
    routed using swap strategies will be added back to the dag taking the ``current_layout``
    into account.
    """

    def __init__(
        self,
        swap_strategy: SwapStrategy | None = None,
        edge_coloring: dict[tuple[int, int], int] | None = None,
    ) -> None:
        r"""
        Args:
            swap_strategy: An instance of a :class:`.SwapStrategy` that holds the swap layers
                that are used, and the order in which to apply them, to map the instruction to
                the hardware. If this field is not given, it should be contained in the
                property set of the pass. This allows other passes to determine the most
                appropriate swap strategy at run-time.
            edge_coloring: An optional edge coloring of the coupling map (I.e. no two edges that
                share a node have the same color). If the edge coloring is given then the commuting
                gates that can be simultaneously applied given the current qubit permutation are
                grouped according to the edge coloring and applied according to this edge
                coloring. Here, a color is an int which is used as the index to define and
                access the groups of commuting gates that can be applied simultaneously.
                If the edge coloring is not given then the sets will be built-up using a
                greedy algorithm. The edge coloring is useful to position gates such as
                ``RZZGate``\s next to swap gates to exploit CX cancellations.
        """
        super().__init__()
        self._swap_strategy = swap_strategy
        self._bit_indices: dict[Qubit, int] | None = None
        self._edge_coloring = edge_coloring

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass by decomposing the nodes it applies on.

        Args:
            dag: The dag to which we will add swaps.

        Returns:
            A dag where swaps have been added for the intended gate type.

        Raises:
            TranspilerError: If the swap strategy was not given at init time and there is
                no swap strategy in the property set.
            TranspilerError: If the quantum circuit contains more than one qubit register.
            TranspilerError: If there are qubits that are not contained in the quantum register.
        """
        if self._swap_strategy is None:
            swap_strategy = self.property_set["swap_strategy"]

            if swap_strategy is None:
                raise TranspilerError("No swap strategy given at init or in the property set.")
        else:
            swap_strategy = self._swap_strategy

        if len(dag.qregs) != 1:
            raise TranspilerError(
                f"{self.__class__.__name__} runs on circuits with one quantum register."
            )

        if len(dag.qubits) != next(iter(dag.qregs.values())).size:
            raise TranspilerError("Circuit has qubits not contained in the qubit register.")

        # Fix output permutation -- copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_dag = dag.copy_empty_like()
        current_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        # Used to keep track of nodes that do not decompose using swap strategies.
        accumulator = new_dag.copy_empty_like()

        for node in dag.topological_op_nodes():
            if isinstance(node.op, Commuting2qBlock):

                # Check that the swap strategy creates enough connectivity for the node.
                self._check_edges(dag, node, swap_strategy)

                # Compose any accumulated non-swap strategy gates to the dag
                accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)

        self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

        self.property_set["virtual_permutation_layout"] = current_layout

        return new_dag

    def _compose_non_swap_nodes(
        self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit
    ) -> DAGCircuit:
        """Add all the non-swap strategy nodes that we have accumulated up to now.

        This method also resets the node accumulator to an empty dag.

        Args:
            layout: The current layout that keeps track of the swaps.
            new_dag: The new dag that we are building up.
            accumulator: A DAG to keep track of nodes that do not decompose
                using swap strategies.

        Returns:
            A new accumulator with the same registers as ``new_dag``.
        """
        # Add all the non-swap strategy nodes that we have accumulated up to now.
        order = layout.reorder_bits(new_dag.qubits)
        order_bits: list[int | None] = [None] * len(layout)
        for idx, val in enumerate(order):
            order_bits[val] = idx

        new_dag.compose(accumulator, qubits=order_bits)

        # Re-initialize the node accumulator
        return new_dag.copy_empty_like()

    def _position_in_cmap(self, dag: DAGCircuit, j: int, k: int, layout: Layout) -> tuple[int, ...]:
        """A helper function to track the movement of virtual qubits through the swaps.

        Args:
            j: The index of decision variable j (i.e. virtual qubit).
            k: The index of decision variable k (i.e. virtual qubit).
            layout: The current layout that takes into account previous swap gates.

        Returns:
            The position in the coupling map of the virtual qubits j and k as a tuple.
        """
        bit0 = dag.find_bit(layout.get_physical_bits()[j]).index
        bit1 = dag.find_bit(layout.get_physical_bits()[k]).index

        return bit0, bit1

    def _build_sub_layers(
        self, current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """A helper method to build-up sets of gates to simultaneously apply.

        This is done with an edge coloring if the ``edge_coloring`` init argument was given or with
        a greedy algorithm if not. With an edge coloring all gates on edges with the same color
        will be applied simultaneously. These sublayers are applied in the order of their color,
        which is an int, in increasing color order.

        Args:
            current_layer: All gates in the current layer can be applied given the qubit ordering
                of the current layout. However, not all gates in the current layer can be applied
                simultaneously. This function creates sub-layers by building up sub-layers
                of gates. All gates in a sub-layer can simultaneously be applied given the coupling
                map and current qubit configuration.

        Returns:
             A list of gate dicts that can be applied. The gates a position 0 are applied first.
             A gate dict has the qubit tuple as key and the gate to apply as value.
        """
        if self._edge_coloring is not None:
            return self._edge_coloring_build_sub_layers(current_layer)
        else:
            return self._greedy_build_sub_layers(current_layer)

    def _edge_coloring_build_sub_layers(
        self, current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """The edge coloring method of building sub-layers of commuting gates."""
        sub_layers: list[dict[tuple[int, int], Gate]] = [
            {} for _ in set(self._edge_coloring.values())
        ]
        # print("CURRENT LAYER", current_layer)
        for edge, gate in current_layer.items():
            color = self._edge_coloring[edge]
            sub_layers[color][edge] = gate

        return sub_layers

    @staticmethod
    def _greedy_build_sub_layers(
        current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """The greedy method of building sub-layers of commuting gates."""
        sub_layers = []
        while len(current_layer) > 0:
            current_sub_layer, remaining_gates = {}, {}
            blocked_vertices: set[tuple] = set()

            for edge, evo_gate in current_layer.items():
                if blocked_vertices.isdisjoint(edge):
                    current_sub_layer[edge] = evo_gate

                    # A vertex becomes blocked once a gate is applied to it.
                    blocked_vertices = blocked_vertices.union(edge)
                else:
                    remaining_gates[edge] = evo_gate

            current_layer = remaining_gates
            sub_layers.append(current_sub_layer)

        return sub_layers

    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: SwapStrategy
    ) -> DAGCircuit:
        """Take an instance of :class:`.Commuting2qBlock` and map it to the coupling map.

        The mapping is done with the swap strategy.

        Args:
            dag: The dag which contains the :class:`.Commuting2qBlock` we route.
            node: A node whose operation is a :class:`.Commuting2qBlock`.
            current_layout: The layout before the swaps are applied. This function will
                modify the layout so that subsequent gates can be properly composed on the dag.
            swap_strategy: The swap strategy used to decompose the node.

        Returns:
            A dag that is compatible with the coupling map where swap gates have been added
            to map the gates in the :class:`.Commuting2qBlock` to the hardware.
        """
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        gate_layers = self._make_op_layers(dag, node.op, current_layout, swap_strategy)
        print("GATE LAYERS", gate_layers)

        # Iterate over and apply gate layers
        max_distance = max(gate_layers.keys())

        circuit_with_swap = QuantumCircuit(len(dag.qubits))

        for i in range(max_distance + 1):
            # Get current layer and replace the problem indices j,k by the corresponding
            # positions in the coupling map. The current layer corresponds
            # to all the gates that can be applied at the ith swap layer.
            current_layer = {}
            for (j, k), local_gate in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(dag, j, k, current_layout)] = local_gate

            # Not all gates that are applied at the ith swap layer can be applied at the same
            # time. We therefore greedily build sub-layers.
            sub_layers = self._build_sub_layers(current_layer)

            # Apply sub-layers
            for sublayer in sub_layers:
                for edge, local_gate in sublayer.items():
                    circuit_with_swap.append(local_gate, edge)

            # Apply SWAP gates
            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]

                    circuit_with_swap.swap(j, k)
                    current_layout.swap(j, k)

        return circuit_to_dag(circuit_with_swap)

    def _make_op_layers(
        self, dag: DAGCircuit, op: Commuting2qBlock, layout: Layout, swap_strategy: SwapStrategy
    ) -> dict[int, dict[tuple, Gate]]:
        """Creates layers of two-qubit gates based on the distance in the swap strategy."""

        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)

        for node in op.node_block:
            edge = (dag.find_bit(node.qargs[0]).index, dag.find_bit(node.qargs[1]).index)

            bit0 = layout.get_virtual_bits()[dag.qubits[edge[0]]]
            bit1 = layout.get_virtual_bits()[dag.qubits[edge[1]]]

            distance = swap_strategy.distance_matrix[bit0, bit1]

            gate_layers[distance][edge] = node.op

        return gate_layers

    def _check_edges(self, dag: DAGCircuit, node: DAGOpNode, swap_strategy: SwapStrategy):
        """Check if the swap strategy can create the required connectivity.

        Args:
            node: The dag node for which to check if the swap strategy provides enough connectivity.
            swap_strategy: The swap strategy that is being used.

        Raises:
            TranspilerError: If there is an edge that the swap strategy cannot accommodate
                and if the pass has been configured to raise on such issues.
        """
        required_edges = set()

        for sub_node in node.op:
            edge = (dag.find_bit(sub_node.qargs[0]).index, dag.find_bit(sub_node.qargs[1]).index)
            required_edges.add(edge)

        # Check that the swap strategy supports all required edges
        if not required_edges.issubset(swap_strategy.possible_edges):
            raise TranspilerError(
                f"{swap_strategy} cannot implement all edges in {required_edges}."
            )


if __name__ == "__main__":
    from networkx import barabasi_albert_graph, draw
    graph = barabasi_albert_graph(n=5, m=3, seed=42)
    from qopt_best_practices.utils import build_max_cut_paulis
    from qiskit.quantum_info import SparsePauliOp

    local_correlators = build_max_cut_paulis(graph)
    cost_operator = SparsePauliOp.from_list(local_correlators)
    print(cost_operator)
    num_qubits = cost_operator.num_qubits
    qaoa_layers = 1
    qaoa_circuit = annotated_qaoa_ansatz(cost_operator, reps = qaoa_layers)
    # qaoa_circuit.measure_all()
    print(qaoa_circuit.draw())

    swap_strategy = SwapStrategy.from_line([i for i in range(num_qubits)])
    edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}
    num_qubits = cost_operator.num_qubits
    print(num_qubits)
    print("EDGE COLORING", edge_coloring)

    pre_init = PassManager(
    [
        PrepareCostLayer(),
        Commuting2qGateRouter(swap_strategy, edge_coloring),
        # SwapToFinalMapping(),  # Removes unnecessary SWAP gates that the end of the block
        # HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
        # InverseCancellation(gates_to_cancel=[CXGate()]),
    ]
    )
    out_circuit = pre_init.run(qaoa_circuit)
    print("AFTER")
    print(out_circuit.draw())

    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    cmap = CouplingMap.from_heavy_hex(distance=3)
    print(cmap.size())
    # backend = GenericBackendV2(num_qubits = 19, coupling_map = cmap, basis_gates = ["x", "sx", "cz", "id", "rz"], seed=0)
    backend = GenericBackendV2(num_qubits = 5, basis_gates = ["x", "sx", "cz", "id", "rz"], seed=0)

    from qiskit.transpiler import generate_preset_pass_manager
    staged_pm = generate_preset_pass_manager(3, backend)
    staged_pm.pre_init = pre_init

    import time
    t0_opt = time.time()
    optimally_transpiled_qaoa = staged_pm.run(qaoa_circuit)
    t1_opt = time.time()

    optimal_count = optimally_transpiled_qaoa.count_ops().get("cz", 0)
    print(f"2q gate count for optimal circuit = {optimal_count}")

    optimal_2q_depth = optimally_transpiled_qaoa.depth(filter_function=lambda x: x.operation.name == "cz")
    print(f"2q depth for optimal circuit = {optimal_2q_depth}")

    optimal_depth = optimally_transpiled_qaoa.depth()
    print(f"total depth for optimal circuit = {optimal_depth}")

    time_optimal = (t1_opt - t0_opt)
    print(f"total time for optimal transpilation = {time_optimal} (s)")

    print(optimally_transpiled_qaoa)

    from qiskit.circuit.library import qaoa_ansatz
    second_qaoa = qaoa_ansatz(cost_operator, reps = qaoa_layers)

    print(second_qaoa.parameters)
    print(optimally_transpiled_qaoa.parameters)
    second_qaoa = second_qaoa.assign_parameters([0.1, 0.2])
    optimally_transpiled_qaoa = optimally_transpiled_qaoa.assign_parameters([0.1, 0.2])
    a = Operator(optimally_transpiled_qaoa)
    print(a)
    b = Operator(second_qaoa)
    print(b)
    print(a==b)
    