"""Utility methods to work with backends."""

from qiskit.transpiler import Target


def remove_instruction_from_target(target: Target, gate_name: str) -> Target:
    """Create a new target without the specified instruction.
    
    Args:
        target: The target to copy and remove the specified instruction.
        gate_name: The name of the instruction to remove.
    """
    # Copy meta information as needed
    new_target = Target(
        description=target.description,
        num_qubits=target.num_qubits,
        dt=target.dt,
        granularity=target.granularity,
        min_length=target.min_length,
        pulse_alignment=target.pulse_alignment,
        acquire_alignment=target.acquire_alignment,
        qubit_properties=target.qubit_properties,
        concurrent_measurements=target.concurrent_measurements,
    )

    # Copy all instructions except the one to remove
    for name, qarg_map in target.items():
        if name == gate_name:
            continue  # Skip the gate you want to remove

        instruction = target.operation_from_name(name)
        new_target.add_instruction(instruction, qarg_map, name=name)

    return new_target
