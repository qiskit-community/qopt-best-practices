# Optimization best practices How-tos

The how-tos in the qopt_best_practices repository will show you how to perform certain tasks to run QAOA-like circuit on quantum hardware.
We currently have the following how-tos in this repository

* 00_qaoa_transpilation_ui.ipynb shows how to transpile QAOA circuits at a high-level using swap strategies and SAT mapping.
* 01_swap_strategies.ipynb explains what swap strategies are and how to use them.
* 02_sat_initial_mapping.ipynb shows how to use the `SATMapper` to optimize the initial mapping of decision variables to qubits to reduce the circuit depth of QAOA.
* 03_how_to_select_qubits.ipynb shows how to select qubits on a backend based on their quality and topology.
* 04_annotated_qaoa_transpilation_details.ipynb shows the details and inner working of the preset pass manager that is used in 00_qaoa_transpilation_ui.ipynb.
