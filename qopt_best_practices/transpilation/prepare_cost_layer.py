"""A pass to format the cost layer for the `Commuting2qGateRouter`."""

from qiskit.transpiler import TransformationPass

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)


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
        commuting_nodes = []
        for node in dag.topological_op_nodes():
            if node.op.name not in ["rz", "rzz"]:
                raise ValueError(
                    f"{self.__class__.__name__} only supports rz and rzz nodes. "
                    f"Found {node.op.name} instead."
                )

            if node.op.name == "rzz":
                commuting_nodes.append(node)

        commuting_block = Commuting2qBlock(commuting_nodes)

        wire_order = {
            wire: idx for idx, wire in enumerate(dag.qubits) if wire not in dag.idle_wires()
        }

        dag.replace_block_with_op(commuting_nodes, commuting_block, wire_order)
