"""Make a pass manager to transpile QAOA."""

from typing import Any, Dict

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)
from qiskit.circuit.library import CXGate

from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


def qaoa_swap_strategy_pm(config: Dict[str, Any]):
    """Provide a pass manager to build the QAOA cirucit.

    This function will be extended in the future.
    """

    num_layers = config.get("num_layers", 1)
    swap_strategy = config.get("swap_strategy", None)
    edge_coloring = config.get("edge_coloring", None)
    basis_gates = config.get("basis_gates", ["sx", "x", "rz", "cx", "id"])

    if swap_strategy is None:
        raise ValueError("No swap_strategy provided in config.")

    if edge_coloring is None:
        raise ValueError("No edge_coloring provided in config.")

    # 2. define pass manager for cost layer
    qaoa_pm = PassManager(
        [
            HighLevelSynthesis(basis_gates=["PauliEvolution"]),
            FindCommutingPauliEvolutions(),
            Commuting2qGateRouter(
                swap_strategy,
                edge_coloring,
            ),
            SwapToFinalMapping(),
            HighLevelSynthesis(basis_gates=basis_gates),
            InverseCancellation(gates_to_cancel=[CXGate()]),
            QAOAConstructionPass(num_layers),
        ]
    )

    return qaoa_pm
