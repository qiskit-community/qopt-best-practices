import json
file = "data/hardware_native_127.json"
data = json.load(open(file, "r"))
paulis = data["paulis"]
num_qubits = len(paulis[0][0])
print(num_qubits)

from qiskit.quantum_info import SparsePauliOp

# define a qiskit SparsePauliOp from the list of paulis
hamiltonian = SparsePauliOp.from_list(paulis)
print(hamiltonian)

from qiskit.circuit.library import QAOAAnsatz

qaoa_circ = QAOAAnsatz(hamiltonian, reps=3)
qaoa_circ.measure_all()
print(qaoa_circ.num_parameters)

from qiskit import transpile
basis_gates = ["rz", "sx", "x", "ecr"]
# Now transpile to sx, rz, x, cx basis
qaoa_circ = transpile(qaoa_circ, basis_gates=basis_gates)
print("transpilation done")

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.get_backend("ibm_nazca")
options = Options()
options.transpilation.skip_transpilation = True
options.execution.shots = 100000

sampler = Sampler(backend=backend, options=options)

import numpy as np

# TQA initialization parameters
dt = 0.75
p = 3  #  3 qaoa layers
grid = np.arange(1, p + 1) - 0.5
init_params = np.concatenate((1 - grid * dt / p, grid * dt / p))
print(init_params)

job = sampler.run(qaoa_circ, init_params)

print(job.job_id())
print(job.result())
