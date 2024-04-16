# q-optimization-best-practices

A collection of guidelines to run quantum optimization algorithms on superconducting qubits with Qiskit,
using as reference the Quantum Approximate Optimization Algorithm (QAOA) workflow.

This repository shows how to combine methods employed in the QAOA literature to get good results on hardware [[1]](https://arxiv.org/abs/2307.14427), 
such as SWAP strategies [[2]](https://arxiv.org/abs/2202.03459), 
SAT mapping [[3]](https://arxiv.org/abs/2212.05666), 
pulse-efficient transpilation and dynamical-decoupling [coming soon]. In the future, it will be expanded 
to include a broader range of quantum algorithms for combinatorial optimization.

The `qopt_best_practices` directory contains a series of reference implementations for the key 
strategies mentioned in the description. These are not intended to be feature-complete, they are
prototypes that should be easy to test and try out in different settings 
(that's why the library is pip-installable!) and can also serve as a guide for your 
own advanced implementations of these techniques. 

The `how-tos` directory contains a series of notebooks you can run to see these techniques in action.
In many cases, the helper methods provided for a specific task (for example, swap mapping), are just 
light wrappers over already-available Qiskit utilities (transpiler passes). How-tos that focus 
on a specific tasks will probably show direct use of the Qiskit code, while how-tos that lay out a full
workflow will likely rely on the provided wrappers for a more general overview of the steps to follow.


This is only a relatively polished repository. If you see any bug or feature you'd like to add, this is a community 
effort, contributions are welcome. Don't hesitate to open an issue or a PR. 

## Quick Start

1. run `git clone https://github.com/qiskit-community/q-optimization-best-practices.git` in your local environment
2. do `pip install -r requirements.txt`
3. do `pip install .`
4. navigate to the `how-tos` section and run the notebook of your choice!


## Table of Contents

The contents of the `qopt_best_practices` package are structured around the key strategies that 
can be applied to best run quantum optimization algorithms on real hardware:

1. Qubit Selection -> `qubit_selection`
2. SAT Mapping -> `sat_mapping`
3. Application of SWAP strategies -> `swap_strategies`
4. QAOA cost function -> `cost_function`

## QAOA Formulation Convention

The convention in this repository is that the (depth-one) QAOA should create an Ansatz of the form:

$$
\begin{align}
\exp\left(-i\beta H_m\right)\exp\left(-i\gamma H_c\right)\vert+\rangle^{\otimes n}
\end{align}
$$

to **minimize** the energy of $H_c$. Since we are minimizing $\\langle H_c\\rangle$, 
the initial state of the ansatz should be the ground state of the mixer $H_m$. This enforces:

$$
\\begin{align}
H_m=-\\sum_iX_i
\\end{align}
$$

As cost operator we apply:

$$
\\begin{align}
H_c=\\sum_{i,j=0}^{n-1}w_{i,j}Z_iZ_j
\\end{align}
$$

where the sum runs over $i < j$.

At the circuit level, these definitions imply that the exponential of the mixer is built 
with $R_x(-2\\beta)$ rotations and the exponential of the cost operator is built from $R_{zz}(2\\gamma w_{i,j})$.


## References
1. Sack, S. H., & Egger, D. J. (2023). Large-scale quantum approximate optimization on non-planar graphs with machine learning noise mitigation. arXiv preprint arXiv:2307.14427. [Link](https://arxiv.org/pdf/2307.14427.pdf).
2. Weidenfeller, J., Valor, L. C., Gacon, J., Tornow, C., Bello, L., Woerner, S., & Egger, D. J. (2022). Scaling of the quantum approximate optimization algorithm on superconducting qubit based hardware. Quantum, 6, 870. [Link](https://arxiv.org/abs/2202.03459).
3. Matsuo, A., Yamashita, S., & Egger, D. J. (2023). A SAT approach to the initial mapping problem in SWAP gate insertion for commuting gates. IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences, 2022EAP1159. [Link](https://arxiv.org/abs/2212.05666).
