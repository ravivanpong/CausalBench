'''
Example for experiment.
'''

import os
import sys
import gc
import time
import concurrent.futures

file_dir = os.path.dirname(__file__)
causalbench_dir = os.path.join(file_dir, '..')
path_result = os.path.join(file_dir, 'result')
sys.path.append(causalbench_dir)

from causalbench.algorithms.gcastle.PC_castle import PC_castle
from causalbench.algorithms.gcastle.ANM_castle import ANM_castle
from causalbench.utils.common import *


# these datasets are from cdt
# list_dataset = ["dream4-1", "dream4-2",
#                 "dream4-3", "dream4-4", "dream4-5", "sachs"]
list_dataset = ["dream4-1"]                
list_standardize = [False, True]
list_algo = [PC_castle(), ANM_castle()]
task_list = combine_multiple_lists([list_algo, list_dataset, list_standardize])

# parameters for algorithem
# could import as json file in the future
dict_algo_param = {
    'castle': {
        'PC': {
            'parameters': {
                'variant': 'stable',
            }
        },
        'ANM': {
            'parameters': {

            }
        }
    }
}


def run(alg, dataset_name, standardize, param):
    # load data
    data, true_graph, true_adj_matrix = load_data_from_cdt(dataset_name)
    if standardize:
        data = standardize_data(data)
    varsort = calc_varsortability(data, true_adj_matrix)

    # estimate
    # print(f"Start to estimate for: \n dataset_name = {dataset_name} \n standardize = {standardize} \n alg = {alg.name}\n")
    starttime = time.perf_counter()
    estimated_adj_matrix = alg.fit(data, param[alg.lib][alg.name])
    finishtime = time.perf_counter()
    runtime = round(finishtime - starttime, 2)

    # evaluate
    shd, shd_cpdag, auc, curve = evaluate_cdt_metrics(
        estimated_adj_matrix, true_graph)

    # output
    dict_result = {
        "dataset_name": dataset_name, "N_variables": data.shape[1], "N_obs": data.shape[0], "standardized": standardize, "varsortability": varsort, "model_name": alg.name, "SHD": float(shd), "SHD_CPAG": float(shd_cpdag), "AOC": 1 - auc, "runtime_second": runtime}
    #print("Summary:\n", dict_result)

    gen_output_file(path_result, "benchmark_result.csv", dict_result)
    gc.collect()


def main():
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for task in task_list:
            executor.submit(run, task[0], task[1], task[2], dict_algo_param)
    finish = time.perf_counter()
    print(f"benchmarking finished in {round(finish - start, 2)} second(s)")


if __name__ == '__main__':
    main()
