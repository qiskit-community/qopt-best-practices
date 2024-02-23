from uuid import uuid4
import json


# auxiliary function to save result data
def save_result(result, backend_type, path="sampler_data"):
    results_to_save = {
        "depth-one": result.quasi_dists[0].binary_probabilities(),
        "depth-zero": result.quasi_dists[1].binary_probabilities(),
    }

    with open(f"{path}/{backend_type}_{str(uuid4())[0:8]}.json", "w") as fout:
        json.dump(results_to_save, fout)
