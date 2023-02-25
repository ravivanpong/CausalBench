import time
import logging
import os
import json
import argparse
import numpy as np
import concurrent.futures
from castle.metrics import MetricsDAG
from causalbench.utils.helper import (
    combine_multiple_lists,
    init_func_with_param,
    load_datasest,
    gen_output_file,
)


def build_full_path(string):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.normpath(os.path.join(script_dir, string))


def init_algo_from_gcastle(
    name: str,
    var_num: int,
    kwargs: dict,
):
    """_summary_

    Args:
        name (str): _description_
        var_num (int): _description_
        kwargs (dict, optional): _description_. Defaults to {}.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if name.lower() == "pc":
        from castle.algorithms import PC

        return init_func_with_param(PC, kwargs)
    elif name.lower() == "ges":
        from castle.algorithms import GES

        return init_func_with_param(GES, kwargs)
    elif name.lower() == "icalingam":
        from castle.algorithms import ICALiNGAM

        return init_func_with_param(ICALiNGAM, kwargs)
    elif name.lower() == "directlingam":
        from castle.algorithms import DirectLiNGAM

        return init_func_with_param(DirectLiNGAM, kwargs)
    elif name.lower() == "anm":  # pairwise
        from castle.algorithms import ANMNonlinear

        return init_func_with_param(ANMNonlinear, kwargs)
    # These algorithms need GPU
    elif name.lower() == "golem":
        from castle.algorithms import GOLEM

        return init_func_with_param(GOLEM, kwargs)
    elif name.lower() == "grandag":
        from castle.algorithms import GraNDAG

        kwargs["input_dim"] = var_num
        return init_func_with_param(GraNDAG, kwargs)
    elif name.lower() == "notears":
        from castle.algorithms import Notears

        return init_func_with_param(Notears, kwargs)
    elif name.lower() == "notearslowrank":
        from castle.algorithms import NotearsLowRank

        return init_func_with_param(NotearsLowRank, kwargs)
    elif name.lower() == "notearsnonlinear":
        from castle.algorithms import NotearsNonlinear

        return init_func_with_param(NotearsNonlinear, kwargs)
    elif name.lower() == "corl":
        from castle.algorithms import CORL

        return init_func_with_param(CORL, kwargs)
    elif name.lower() == "rl":
        from castle.algorithms import RL

        return init_func_with_param(RL, kwargs)
    elif name.lower() == ("gae"):
        from castle.algorithms import GAE

        return init_func_with_param(GAE, kwargs)
    elif name.lower() == ("pnl"):
        from castle.algorithms import PNL

        return init_func_with_param(PNL, kwargs)
    else:
        raise ValueError("Unknown algorithm.")


def run(
    algo_name: str,
    algo_kwargs: dict,
    dataset_name: str,
    dataset_kwargs: dict,
    path_result: str,
    output_file_name: str,
):
    # load data set
    dataset = load_datasest(dataset_name, dataset_kwargs)
    true_causal_matrix = dataset["true_matrix"]
    X = dataset["X"]
    logging.info("%s loaded", dataset["name"])
    # init algorithm
    var_num = dataset["var_num"]
    algo = init_algo_from_gcastle(algo_name, var_num, algo_kwargs)
    logging.info("%s algorithm initiated.", algo_name)
    is_success = True
    err_message = None
    # structure learning
    if (
        algo_name.lower() != "notearslowrank"
    ):  # notearslowrank takes additional argument "rank" for lean method.
        starttime = time.perf_counter()
        try:
            algo.learn(X)
        except Exception as err:
            logging.warning("Error: %s", err)
            is_success = False
            err_message = err
    else:
        rank = np.linalg.matrix_rank(dataset["true_matrix"])
        starttime = time.perf_counter()
        try:
            algo.learn(X, rank)
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
                "library_name": "gCastle",
                "Error": err_message,
            },
        )

    else:
        finishtime = time.perf_counter()
        logging.info("%s on %s done.", algo_name, dataset["name"])
        runtime = round(finishtime - starttime, 2)

        # calculate metrics
        mt = MetricsDAG(algo.causal_matrix, true_causal_matrix)

        dict_result = {
            "dataset_name": dataset["name"],
            "varsortability": dataset["varsortability"],
            "N_variables": X.shape[1],
            "N_obs": X.shape[0],
            "algo_name": algo_name.lower(),
            "algo_param": algo_kwargs,
            "library_name": "gCastle",
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
    os.environ["CASTLE_BACKEND"] = "pytorch"
    algorithms = params["algorithms"][0]["gcastle"]
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
