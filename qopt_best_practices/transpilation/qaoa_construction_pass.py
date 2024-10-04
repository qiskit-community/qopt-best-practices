"""A pass to build a full QAOA ansatz circuit."""

from typing import Optional

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass


class QAOAConstructionPass(TransformationPass):
    """Build the QAOAAnsatz from a transpiled cost operator.

    This pass takes as input a single layer of a transpiled QAOA operator.
    It then repeats this layer the appropriate number of time and adds (i)
    the initial state, (ii) the mixer operator, and (iii) the measurements.
    The measurements are added in such a fashion so as to undo the local
    permuttion induced by the SWAP gates in the cost layer.
    """

    def __init__(
        self,
        num_layers: int,
        init_state: Optional[QuantumCircuit] = None,
        mixer_layer: Optional[QuantumCircuit] = None,
    ):
        """Initialize the pass

        Limitations: The current implementation of the pass does not permute the mixer.
        Therefore mixers with local bias fields, such as in warm-start methods, will not
        result in the correct circuits.

        Args:
            num_layers: The number of QAOA layers to apply.
            init_state: The initial state to use. This must match the anticipated number
                of qubits otherwise an error will be raised when `run` is called. If this
                is not given we will default to the equal superposition initial state.
            mixer_layer: The mixer layer to use. This must match the anticipated number
                of qubits otherwise an error will be raised when `run` is called. If this
                is not given we default to the standard mixer made of X gates.
        """
        super().__init__()

        self.num_layers = num_layers
        self.init_state = init_state
        self.mixer_layer = mixer_layer

    def run(self, cost_layer_dag: DAGCircuit):
        num_qubits = cost_layer_dag.num_qubits()

        # Make the initial state and the mixer.
        if self.init_state is None:
            init_state = QuantumCircuit(num_qubits)
            init_state.h(range(num_qubits))
        else:
            init_state = self.init_state

        if self.mixer_layer is None:
            mixer_layer = QuantumCircuit(num_qubits)
            beta = Parameter("β")
            mixer_layer.rx(2 * beta, range(num_qubits))
        else:
            mixer_layer = self.mixer_layer

        # Do some sanity checks on qubit numbers.
        if init_state.num_qubits != num_qubits:
            raise TranspilerError(
                "Number of qubits in the initial state does not match the number in the DAG. "
                f"{init_state.num_qubits} != {num_qubits}"
            )

        if mixer_layer.num_qubits != num_qubits:
            raise TranspilerError(
                "Number of qubits in the mixer does not match the number in the DAG. "
                f"{init_state.num_qubits} != {num_qubits}"
            )

        cost_layer = dag_to_circuit(cost_layer_dag)
        qaoa_circuit = QuantumCircuit(num_qubits, num_qubits)

        # Re-parametrize the circuit
        gammas = ParameterVector("γ", self.num_layers)
        betas = ParameterVector("β", self.num_layers)

        # Add initial state
        qaoa_circuit.compose(init_state, inplace=True)

        # iterate over number of qaoa layers
        # and alternate cost/reversed cost and mixer
        for layer in range(self.num_layers):
            bind_dict = {cost_layer.parameters[0]: gammas[layer]}
            bound_cost_layer = cost_layer.assign_parameters(bind_dict)

            bind_dict = {mixer_layer.parameters[0]: betas[layer]}
            bound_mixer_layer = mixer_layer.assign_parameters(bind_dict)

            if layer % 2 == 0:
                # even layer -> append cost
                qaoa_circuit.compose(bound_cost_layer, range(num_qubits), inplace=True)
            else:
                # odd layer -> append reversed cost
                qaoa_circuit.compose(
                    bound_cost_layer.reverse_ops(), range(num_qubits), inplace=True
                )

            # the mixer layer is not reversed and not permuted.
            qaoa_circuit.compose(bound_mixer_layer, range(num_qubits), inplace=True)

        if self.num_layers % 2 == 1:
            # iterate over layout permutations to recover measurements
            if self.property_set["virtual_permutation_layout"]:
                for cidx, qidx in (
                    self.property_set["virtual_permutation_layout"].get_physical_bits().items()
                ):
                    qaoa_circuit.measure(qidx, cidx)
            else:
                print("layout not found, assigining trivial layout")
                for idx in range(num_qubits):
                    qaoa_circuit.measure(idx, idx)
        else:
            for idx in range(num_qubits):
                qaoa_circuit.measure(idx, idx)

        return circuit_to_dag(qaoa_circuit)
