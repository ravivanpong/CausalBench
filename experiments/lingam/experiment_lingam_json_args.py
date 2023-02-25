import time
import logging
import os
import json
import argparse
import concurrent.futures
import numpy as np
from castle.metrics import MetricsDAG
from lingam import (
    DirectLiNGAM,
    ICALiNGAM,
    MultiGroupDirectLiNGAM,
    VARLiNGAM,
    VARMALiNGAM,
    LongitudinalLiNGAM,
)
from causalbench.utils.helper import (
    combine_multiple_lists,
    init_func_with_param,
    load_datasest,
    gen_output_file,
)


def build_full_path(string):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.normpath(os.path.join(script_dir, string))


def init_algo_from_lingam(name: str, kwargs: dict):
    """

    Args:
        name (str): _description_
        kwargs (dict): _description_
    """
    if name.lower() == "directlingam":
        return init_func_with_param(DirectLiNGAM, kwargs)
    elif name.lower() == "icalingam":
        return init_func_with_param(ICALiNGAM, kwargs)
    elif name.lower() == "multigroupdirectlingam":
        return init_func_with_param(MultiGroupDirectLiNGAM, kwargs)
    elif name.lower() == "varlingam":
        return init_func_with_param(VARLiNGAM, kwargs)
    elif name.lower() == "varmalingam":
        return init_func_with_param(VARMALiNGAM, kwargs)
    elif name.lower() == "longitudinallingam":
        return init_func_with_param(LongitudinalLiNGAM, kwargs)
    # LIM, bottomupparcelingam, CAM-UV, rcd, lina, mdlina, resit are taged as pairwise
    else:
        raise ValueError("Unknown algorithm.")


def run(
    algo_name: str,
    algo_kwargs: dict,
    dataset_name: str,
    dataset_kwargs: dict,
    path_result: str,
    output_file_name: str,
    nan_as_one: True,
):
    # load data set
    dataset = load_datasest(dataset_name, dataset_kwargs)
    true_causal_matrix = dataset["true_matrix"]
    X = dataset["X"]
    logging.info("%s loaded", dataset["name"])
    # init algorithm
    algo = init_algo_from_lingam(algo_name, algo_kwargs)
    logging.info("%s algorithm initiated.", algo_name)
    is_success = True
    err_message = None
    # structure learning

    starttime = time.perf_counter()
    try:
        algo.fit(X)
    except Exception as err:
        logging.warning("Error: %s", err)
        is_success = False
        err_message = err

    if not is_success:
        gen_output_file(
            path_result,
            f"{output_file_name}.csv",
            {
                "dataset_name": dataset["name"],
                "varsortability": dataset["varsortability"],
                "N_variables": X.shape[1],
                "N_obs": X.shape[0],
                "algo_name": algo_name.lower(),
                "algo_param": algo_kwargs,
                "library_name": "LiNGAM",
                "Error": err_message,
            },
        )

    else:
        # if nan_as_one then set all nan in est_causal_matrix to 1 else set them to 0
        estimated_causal_matrix = (
            np.where(algo.adjacency_matrix_ == 0, 0, 1)
            if nan_as_one
            else (abs(algo.adjacency_matrix_) > 0).astype(int)
        )
        finishtime = time.perf_counter()
        logging.info("%s on %s done.", algo_name, dataset_name)
        runtime = round(finishtime - starttime, 2)

        # calculate metrics
        mt = MetricsDAG(estimated_causal_matrix, true_causal_matrix)

        dict_result = {
            "dataset_name": dataset["name"],
            "varsortability": dataset["varsortability"],
            "N_variables": X.shape[1],
            "N_obs": X.shape[0],
            "algo_name": algo_name.lower(),
            "algo_param": algo_kwargs,
            "library_name": "LiNGAM",
            "fdr": mt.metrics["fdr"],
            "tpr": mt.metrics["tpr"],
            "fpr": mt.metrics["fpr"],
            "shd": mt.metrics["shd"],
            "nnz": mt.metrics["nnz"],
            "precision": mt.metrics["precision"],
            "recall": mt.metrics["recall"],
            "F1": mt.metrics["F1"],
            "gscore": mt.metrics["gscore"],
            "runtime_second": runtime,
            "experiment_time": time.ctime(),
        }
        gen_output_file(path_result, f"{output_file_name}.csv", dict_result)


def main():
    algorithms = params["algorithms"][1]["lingam"]
    algos_to_print = list({algo["name"] for algo in algorithms})
    logging.info("algorithms are: %s", algos_to_print)
    datasets = params["datasets"]
    datasets_to_print = list({dataset["name"] for dataset in datasets})
    logging.info("datasets are: %s", datasets_to_print)
    output_file_name = params["OUTPUT_FILE_NAME"]
    # set output file path
    file_dir = os.path.dirname(__file__)
    path_result = os.path.join(file_dir, "result")
    try:
        os.mkdir(path_result)
    except FileExistsError:
        logging.info("result dir already exists")
    else:
        logging.info("result dir created.")

    tasks = combine_multiple_lists([algorithms, datasets])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for task in tasks:
            executor.submit(
                run,
                task[0]["name"],
                task[0]["kwargs"],
                task[1]["name"],
                task[1]["kwargs"],
                path_result,
                output_file_name,
                nan_as_one=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalBench")
    parser.add_argument("-src", required=True, help="relative path to parameter file")
    param_file_path = build_full_path(parser.parse_args().src)
    logging.info("file to read: %s", param_file_path)

    params = None
    with open(param_file_path) as json_file:
        params = json.load(json_file)

    main()
