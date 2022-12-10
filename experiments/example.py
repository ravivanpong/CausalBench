"""
Example experiment.
"""
import os
import gc
import time
import logging

import concurrent.futures
import causalbench.utils.common as util

from causalbench.algorithms.gcastle.anm_castle import ANMCastle
from causalbench.algorithms.gcastle.pc_castle import PCCastle


# these datasets are from cdt
# list_dataset = ["dream4-1", "dream4-2",
#                 "dream4-3", "dream4-4", "dream4-5", "sachs"]
list_dataset = ["dream4-1"]
list_standardize = [False, True]
list_algo = [PCCastle(variant="stable"), ANMCastle()]
task_list = util.combine_multiple_lists([list_algo, list_dataset, list_standardize])

# parameters for algorithm
# could import as json file in the future
# dict_algo_param = {
#     "castle": {
#         "PC": {
#             "parameters": {
#                 "variant": "stable",
#             }
#         },
#         "ANM": {"parameters": {}},
#     }
# }

file_dir = os.path.dirname(__file__)

path_result = os.path.join(file_dir, "result")
try:
    os.mkdir(path_result)
except FileExistsError:
    logging.info("result dir already exists")
else:
    logging.info("result dir created.")


def run(alg, dataset_name, standardize):
    """_summary_

    Args:
        alg (_type_): _description_
        dataset_name (_type_): _description_
        standardize (_type_): _description_
        param (_type_): _description_
    """
    # load data
    data, true_graph, true_adj_matrix = util.load_data_from_cdt(dataset_name)
    data = util.standardize_data(data) if standardize else data
    varsort = util.calc_varsortability(data, true_adj_matrix)

    # estimate
    starttime = time.perf_counter()
    estimated_adj_matrix = alg.fit(data)
    finishtime = time.perf_counter()
    runtime = round(finishtime - starttime, 2)

    # evaluate
    shd, shd_cpdag, auc, *_ = util.evaluate_cdt_metrics(
        estimated_adj_matrix, true_graph
    )

    # output
    dict_result = {
        "dataset_name": dataset_name,
        "N_variables": data.shape[1],
        "N_obs": data.shape[0],
        "standardized": standardize,
        "varsortability": varsort,
        "model_name": alg.name,
        "SHD": float(shd),
        "SHD_CPAG": float(shd_cpdag),
        "AOC": 1 - auc,
        "runtime_second": runtime,
        "experiment_time": time.ctime(),
    }
    # print("Summary:\n", dict_result)

    util.gen_output_file(path_result, "example_result.csv", dict_result)
    gc.collect()


def main():
    """_summary_"""
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for task in task_list:
            executor.submit(run, task[0], task[1], task[2])
    finish = time.perf_counter()
    print(f"benchmarking finished in {round(finish - start, 2)} second(s)")


# no multi-threading
# def main():
#     """_summary_"""
#     start = time.perf_counter()
#     for task in task_list:
#         run(task[0], task[1], task[2])
#     finish = time.perf_counter()
#     print(f"benchmarking finished in {round(finish - start, 2)} second(s)")


if __name__ == "__main__":
    main()
