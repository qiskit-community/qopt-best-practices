import json
file = "data/hardware_native_127.json"
data = json.load(open(file, "r"))
paulis = data["paulis"]
num_qubits = len(paulis[0][0])
print("Num qubits", num_qubits)

from qiskit.quantum_info import SparsePauliOp

# define a qiskit SparsePauliOp from the list of paulis
hamiltonian = SparsePauliOp.from_list(paulis)
print("Hamiltonian", hamiltonian)

from qiskit.circuit.library import QAOAAnsatz

qaoa_circ = QAOAAnsatz(hamiltonian, reps=3)
qaoa_circ.measure_all()
print("Num params", qaoa_circ.num_parameters)

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options, Session
from tokens import TOKEN

# QiskitRuntimeService.save_account(channel="ibm_quantum", token=TOKEN, overwrite=True)
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.get_backend("ibm_nazca")
options = Options()
options.resilience_level = 0
options.transpilation.skip_transpilation = True
options.execution.shots = 100
print("100 shots, resilience level 0, skip transpilation True")

from qiskit import transpile
# Now transpile to backend
qaoa_circ = transpile(qaoa_circ, backend=backend)
print("Transpilation done")

import numpy as np

# TQA initialization parameters
dt = 0.75
p = 3  #  3 qaoa layers
grid = np.arange(1, p + 1) - 0.5
init_params = np.concatenate((1 - grid * dt / p, grid * dt / p))
print("TQA init params",init_params)

from qopt_best_practices.cost_function import evaluate_sparse_pauli


def cost_func_sampler(params, ansatz, hamiltonian, sampler):
    
    t1 = time.time()
    job = sampler.run(ansatz, params)
    sampler_result = job.result()
    print("New iter, time taken sampler:", time.time()-t1) 
    sampled = sampler_result.quasi_dists[0]

    # a dictionary containing: {state: (measurement probability, value)}
    evaluated = {
        state: (probability, evaluate_sparse_pauli(state, hamiltonian))
        for state, probability in sampled.items()
    }

    result = sum(probability * value for probability, value in evaluated.values())
    print("Iteration result:", result)
    return result

from scipy.optimize import minimize

print("Running minimization..")
import time

with Session(backend=backend):
    sampler = Sampler(options=options)
    t0 = time.time()
    result = minimize(
        cost_func_sampler,
        init_params,
        args=(qaoa_circ, hamiltonian, sampler),
        method="COBYLA",
        options={"maxiter":3},
    )
    print("Final result", result)
    print("Minimization time", time.time()-t0)


