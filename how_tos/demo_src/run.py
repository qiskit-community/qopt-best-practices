from qopt_best_practices.cost_function import evaluate_sparse_pauli
from scipy.optimize import minimize
import numpy as np
from dataclasses import dataclass

def evaluate_cost(sampled, hamiltonian):

    # a dictionary containing: {state: (measurement probability, value)}
    evaluated = {
        state: (probability, evaluate_sparse_pauli(state, hamiltonian))
        for state, probability in sampled.items()
    }

    result = sum(probability * value for probability, value in evaluated.values())

    return np.real(result)

@dataclass
class OptResult():
    message = None
    param_values = None
    cost_function_value = None
    num_iter = None
    
def run_minimization_loop(cost_function, circuit, hamiltonian, sampler):
    # define initial point
    initial_point = 2 * np.pi * np.random.rand(circuit.num_parameters)
    # minimize
    result = minimize(
        cost_function,
        initial_point,
        args=(circuit, hamiltonian, sampler),
        method="COBYLA",
    )

    opt_result = OptResult()
    opt_result.message = result.message
    opt_result.param_values = result.x
    opt_result.cost_function_value = result.fun
    opt_result.num_iter = result.nfev
    
    return opt_result