"""Pass to remove SWAP gates that are not needed."""

from qiskit.dagcircuit import DAGOutNode, DAGCircuit
from qiskit.transpiler import TransformationPass


class SwapToFinalMapping(TransformationPass):
    """Absorb any redundent SWAPs in the final layout.

    This pass should be executed after a SWAPStrategy has been applied to a block
    of commuting gates. It will remove any final redundent SWAP gates and absorb
    them into the virtual layout. This effectively undoes any possibly redundent
    SWAP gates that the SWAPStrategy may have inserted.
    """

    def run(self, dag: DAGCircuit):
        """run the pass."""

        qmap = self.property_set["virtual_permutation_layout"]

        qreg = dag.qregs["q"]

        # This will remove SWAP gates that are applied before anything else
        # This remove is executed multiple times until there are no more SWAP
        # gates left to remove. Note: a more inteligent DAG traversal could
        # be implemented here.

        done = False

        while not done:
            permuted = False
            for node in dag.topological_op_nodes():
                if node.op.name == "swap":
                    successors = list(dag.successors(node))
                    if len(successors) == 2:
                        if all(isinstance(successors[idx], DAGOutNode) for idx in [0, 1]):
                            bits = [qreg.index(qubit) for qubit in node.qargs]
                            qmap[bits[0]], qmap[bits[1]] = qmap[bits[1]], qmap[bits[0]]
                            dag.remove_op_node(node)
                            permuted = True

            done = not permuted

        return dag
