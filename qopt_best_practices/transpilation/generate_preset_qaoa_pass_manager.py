"""Make a pass manager to transpile QAOA."""

from typing import Dict, Tuple

from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.transpiler import Layout, PassManager
from qiskit.transpiler.passes import BasisTranslator, UnrollCustomDefinitions

from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qopt_best_practices.transpilation.annotated_transpilation_passes import (
    AnnotatedPrepareCostLayer,
    AnnotatedCommuting2qGateRouter,
    AnnotatedSwapToFinalMapping,
    SynthesizeAndSimplifyCostLayer,
    UnrollBoxes,
)


def generate_preset_qaoa_pass_manager(
    backend,
    swap_strategy: SwapStrategy,
    edge_coloring: Dict[Tuple[int, int], int] = None,
    initial_layout: Layout = None,
):
    """Provide a pass manager to build the QAOA cirucit.

    This function will be extended in the future.
    """

    # 1. define pass manager for annotated qaoa ansatz
    pre_init = PassManager(
        [
            AnnotatedPrepareCostLayer(),
            AnnotatedCommuting2qGateRouter(swap_strategy, edge_coloring),
            AnnotatedSwapToFinalMapping(),
            SynthesizeAndSimplifyCostLayer(basis_gates=["x", "cx", "sx", "rz", "id"]),
            UnrollBoxes(),
        ]
    )

    # 2. The post init step unrolls the gates in the ansatz to the backend basis gates
    post_init = PassManager(
        [
            UnrollCustomDefinitions(_sel, basis_gates=backend.operation_names, min_qubits=3),
            BasisTranslator(_sel, target_basis=backend.operation_names, min_qubits=3),
        ]
    )

    staged_pm = generate_preset_pass_manager(
        3,
        backend,
        initial_layout=initial_layout,
    )

    staged_pm.pre_init = pre_init
    staged_pm.post_init = post_init

    return staged_pm
