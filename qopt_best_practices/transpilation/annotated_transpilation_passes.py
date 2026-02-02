"""Annotated QAOA transpilation passes"""

from __future__ import annotations
from collections import defaultdict


from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
)
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qiskit.dagcircuit import DAGOutNode, DAGCircuit, DAGOpNode
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library import SwapGate, Measure
from qiskit.converters import dag_to_circuit, circuit_to_dag


class AnnotatedPrepareCostLayer(TransformationPass):
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
        new_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            if node.op.name == "box":
                if "cost_layer" in node.op.annotations[0].namespace:
                    box_dag = circuit_to_dag(node.op.params[0])
                    # Why does the inner dag not have qregs?
                    # In the meantime, let's add them
                    for qreg in dag.qregs.values():
                        box_dag.add_qreg(qreg)

                    commuting_nodes = []
                    rz_gates = []
                    for box_node in box_dag.topological_op_nodes():
                        if box_node.op.name == "rzz":
                            commuting_nodes.append(box_node)
                        elif box_node.op.name == "rz":
                            rz_gates.append(box_node)
                            box_dag.remove_op_node(box_node)
                        else:
                            raise ValueError(
                                f"{self.__class__.__name__} only supports rz and rzz nodes. "
                                f"Found {box_node.op.name} instead."
                            )

                    commuting_block = Commuting2qBlock(commuting_nodes)

                    wire_order = {
                        wire: idx
                        for idx, wire in enumerate(box_dag.qubits)
                        if wire not in box_dag.idle_wires()
                    }

                    box_dag.replace_block_with_op(commuting_nodes, commuting_block, wire_order)

                    for z_node in rz_gates:
                        box_dag.apply_operation_back(
                            z_node.op, qargs=z_node.qargs, cargs=z_node.cargs
                        )

                    node.op.params[0] = dag_to_circuit(box_dag)
            new_dag.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
        return new_dag


class AnnotatedCommuting2qGateRouter(TransformationPass):
    """A class to swap route one or more commuting gates to the coupling map."""

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

        if not dag.cregs:
            for qreg in dag.qregs.values():
                creg = ClassicalRegister(len(qreg), f"c_{qreg.name}")
                dag.add_creg(creg)

        for node in dag.topological_op_nodes():
            if node.op.name == "box":
                if "cost_layer" in node.op.annotations[0].namespace:
                    box_dag = circuit_to_dag(node.op.params[0])
                    # Fix output permutation -- copied from ElidePermutations
                    input_qubit_mapping = {
                        qubit: index for index, qubit in enumerate(box_dag.qubits)
                    }
                    self.property_set["original_layout"] = Layout(input_qubit_mapping)
                    if self.property_set["original_qubit_indices"] is None:
                        self.property_set["original_qubit_indices"] = input_qubit_mapping

                    new_dag = box_dag.copy_empty_like()
                    current_layout = Layout.generate_trivial_layout(*box_dag.qregs.values())
                    # Used to keep track of nodes that do not decompose using swap strategies.
                    accumulator = new_dag.copy_empty_like()

                    for box_node in box_dag.topological_op_nodes():
                        if isinstance(box_node.op, Commuting2qBlock):
                            # Check that the swap strategy creates enough connectivity for the node.
                            self._check_edges(box_dag, box_node, swap_strategy)
                            # Compose any accumulated non-swap strategy gates to the dag
                            accumulator = self._compose_non_swap_nodes(
                                accumulator, current_layout, new_dag
                            )
                            # Decompose the swap-strategy node and add to the dag.
                            out_decompose = self.swap_decompose(
                                box_dag, box_node, current_layout, swap_strategy
                            )
                            new_dag.compose(out_decompose)
                        else:
                            accumulator.apply_operation_back(
                                box_node.op, box_node.qargs, box_node.cargs
                            )
                    self._compose_non_swap_nodes(accumulator, current_layout, new_dag)
                    self.property_set["virtual_permutation_layout"] = current_layout

                    # Reverse operations on even layers
                    if int(node.op.annotations[0].payload) % 2 != 0:
                        node.op.params[0] = dag_to_circuit(new_dag)
                    else:
                        node.op.params[0] = dag_to_circuit(new_dag).reverse_ops()

                    # Permute final measurements
                    measure_nodes = [
                        node for node in dag.op_nodes() if isinstance(node.op, Measure)
                    ]
                    qmap = self.property_set["virtual_permutation_layout"]
                    if len(measure_nodes) > 0:
                        for m_node in measure_nodes:
                            dag.remove_op_node(m_node)
                        for cidx in range(dag.num_qubits()):
                            qubit = (
                                qmap.get_physical_bits().get(cidx, cidx)
                                if int(node.op.annotations[0].payload) % 2 == 1
                                else dag.qubits[cidx]
                            )
                            dag.apply_operation_back(Measure(), [qubit], [dag.clbits[cidx]])

        return dag

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

        # Iterate over and apply gate layers
        max_distance = max(gate_layers.keys())
        dag_with_swap = DAGCircuit.copy_empty_like(dag)
        for i in range(max_distance + 1):
            current_layer = {}
            for (j, k), local_gate in gate_layers.get(i, {}).items():
                edge = self._position_in_cmap(dag, j, k, current_layout)
                current_layer[edge] = local_gate

            sub_layers = self._build_sub_layers(current_layer)

            for sublayer in sub_layers:
                for edge, local_gate in sublayer.items():
                    qubit_edge = [dag_with_swap.qubits[i] for i in edge]
                    dag_with_swap.apply_operation_back(local_gate, qubit_edge)

            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]
                    dag_with_swap.apply_operation_back(SwapGate(), [j, k])
                    current_layout.swap(j, k)

        return dag_with_swap

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


class AnnotatedSwapToFinalMapping(TransformationPass):
    """Absorb any redundent SWAPs in the final layout. Add permuted measurements.

    This pass should be executed after a SWAPStrategy has been applied to a block
    of commuting gates. It will remove any final redundent SWAP gates and absorb
    them into the virtual layout. This effectively undoes any possibly redundent
    SWAP gates that the SWAPStrategy may have inserted.
    """

    def run(self, dag: DAGCircuit):
        qmap = self.property_set["virtual_permutation_layout"]
        permuted_cost_layer = None
        num_layers = 0

        for node in dag.topological_op_nodes():
            if node.op.name != "box":
                continue

            annotation = node.op.annotations[0]
            layer_index = int(annotation.payload)
            layer_name = annotation.namespace
            num_layers = max(num_layers, layer_index)

            box_circuit = node.op.params[0]
            box_dag = circuit_to_dag(box_circuit)
            if "cost_layer" in layer_name:
                if layer_index == 1:
                    # Remove final SWAPs and update virtual layout
                    while True:
                        swaps_removed = False
                        for box_node in box_dag.topological_op_nodes():
                            if box_node.op.name == "swap":
                                successors = list(box_dag.successors(box_node))
                                if all(isinstance(s, DAGOutNode) for s in successors):
                                    qmap.swap(box_node.qargs[0], box_node.qargs[1])
                                    box_dag.remove_op_node(box_node)
                                    swaps_removed = True
                        if not swaps_removed:
                            break
                    permuted_cost_layer = box_dag
                    node.op.params[0] = dag_to_circuit(permuted_cost_layer)
                elif permuted_cost_layer:
                    final_params = list(box_circuit.parameters)
                    new_circuit = dag_to_circuit(permuted_cost_layer).assign_parameters(
                        final_params
                    )
                    new_dag = circuit_to_dag(
                        new_circuit.reverse_ops() if layer_index % 2 == 0 else new_circuit
                    )
                    node.op.params[0] = dag_to_circuit(new_dag)
                else:
                    raise TypeError("Missing permuted cost layer for substitution.")

            elif "mixer" in layer_name:
                if layer_index % 2 == 1:
                    # Permute mixer layer according to virtual permutation layout
                    new_dag = box_dag.copy_empty_like()
                    inverse_layout = qmap.get_physical_bits()
                    for box_node in box_dag.op_nodes():
                        new_qargs = []
                        if len(box_node.qargs) > 1:
                            raise NotImplementedError(
                                "Routing for two-local mixers is not yet supported."
                            )
                        for qubit in box_node.qargs:
                            physical_index = inverse_layout[qubit._index]
                            new_qargs.append(new_dag.qubits[physical_index._index])

                        new_dag.apply_operation_back(box_node.op, qargs=new_qargs)

                    node.op.params[0] = dag_to_circuit(new_dag)

        # Permute final measurements
        measure_nodes = [node for node in dag.op_nodes() if isinstance(node.op, Measure)]

        if len(measure_nodes) > 0:
            for node in measure_nodes:
                dag.remove_op_node(node)
            for cidx in range(dag.num_qubits()):
                qubit = (
                    qmap.get_physical_bits().get(cidx, cidx)
                    if num_layers % 2 == 1
                    else dag.qubits[cidx]
                )
                dag.apply_operation_back(Measure(), [qubit], [dag.clbits[cidx]])
        return dag


class UnrollBoxes(TransformationPass):
    """Remove Boxes."""

    def run(self, dag: DAGCircuit):

        for node in dag.topological_op_nodes():
            if node.op.name != "box":
                continue

            box_circuit = node.op.params[0]
            box_dag = circuit_to_dag(box_circuit)
            dag.substitute_node_with_dag(node, box_dag)
        return dag


class SynthesizeAndSimplifyCostLayer(TransformationPass):
    """Decompose cost layer using HLS (High Level Synthesis), apply inverse cancelation."""

    requires = []
    preserves = []

    def __init__(self, target=None, basis_gates=None):

        super().__init__()

        # Unify behavior for target and basis gates inputs
        gate_names = set()
        if basis_gates:
            gate_names.update(basis_gates)
        elif target:
            gate_names.update(target.operation_names)
        two_qubit_gates = []
        for name in gate_names:
            gate = get_standard_gate_name_mapping()[name]
            if gate is not None and getattr(gate, "num_qubits", None) == 2:
                two_qubit_gates.append(gate)

        self.hls = HighLevelSynthesis(target=target, basis_gates=basis_gates)
        self.inverse_cancel = InverseCancellation(gates_to_cancel=two_qubit_gates)

    def run(self, dag: DAGCircuit):
        for node in dag.topological_op_nodes():
            if node.op.name != "box":
                continue
            annotation = node.op.annotations[0]
            layer_name = annotation.namespace
            if "cost_layer" in layer_name:
                box_circuit = node.op.params[0]
                box_dag = circuit_to_dag(box_circuit)
                out_dag = self.inverse_cancel.run(self.hls.run(box_dag))
                node.op.params[0] = dag_to_circuit(out_dag)
        return dag
