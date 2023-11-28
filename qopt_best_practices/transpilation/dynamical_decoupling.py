
from typing import Dict, Optional

from qiskit.transpiler import PassManager
from qiskit.circuit.library import XGate, YGate

from qiskit_ibm_provider.transpiler.passes.scheduling import (
    DynamicCircuitInstructionDurations,
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    PadDelay,
)


PASSES = {
    "staggered_XY4": {
        "spacings": [0.125, 0.25, 0.25, 0.25, 0.125],
        "alt_spacings": [0.25, 0.25, 0.25, 0.25, 0],
        "dd_sequences": [XGate(), YGate(), XGate(), YGate()],
    },
    "staggered_XX": {
        "spacings": [0.25, 0.5, 0.25],
        "alt_spacings": [0.5, 0.5, 0],
        "dd_sequences": [XGate(), XGate()],
    },
    "XY4": {
        "spacings": [0.125, 0.25, 0.25, 0.25, 0.125],
        "alt_spacings": None,  # TODO Check
        "dd_sequences": [XGate(), YGate(), XGate(), YGate()],
    },
    "XX": {
        "spacings": [0.25, 0.5, 0.25],
        "alt_spacings": None,  # TODO Check
        "dd_sequences": [XGate(), XGate()],
    },
}


def dd_pass_manager(
    backend,
    dd_sequence: Optional[str] = None,
    dd_config: Optional[Dict] = None,
):
    """Make a pass-manager that inserts a Dynamical Decoupling sequence into a circuit.

    This is a helper function to make a dynamical decoupling PassManager.

    Args:
        backend: The backend for which to generate the DD sequence. This is needed
            because we need to know the duration of the instructions.
        dd_sequence: A string to specify the DD sequence to use. This corresponds to preset
            values for the gates to insert, the spacings between the gates, as well as the
            `alt_spacings` that can be leveraged for staggered DD.
        dd_config: If users do not want to use a preset DD configuration then they can
            specify the arguments to `PadDynamicalDecoupling` themselves. The config should
            include the arguments `dd_sequences`, `spacings`, and `alt_spacings`.
    """
    min_seq_ratio = 1

    durations = DynamicCircuitInstructionDurations.from_backend(backend)

    # Some backends do not report the duration of the Y gate.
    phys_qs = list(range(backend.num_qubits))
    durations.update(DynamicCircuitInstructionDurations(
        [("y", i, durations.get('x', i)) for i in phys_qs])
    )

    if dd_config is None:
        dd_config = PASSES.get(dd_sequence, None)

    if dd_config is None:
        raise ValueError(f"Unknown DD sequence. Options for `dd_sequence` are {PASSES.keys()}.")

    return PassManager(
        [
            ALAPScheduleAnalysis(durations),
            PadDynamicalDecoupling(
                durations,
                dd_sequences=dd_config["dd_sequences"],
                coupling_map=backend.coupling_map,
                spacings=dd_config["spacings"],
                alt_spacings=dd_config["alt_spacings"],
                sequence_min_length_ratios=min_seq_ratio,
            )
        ]
    )

