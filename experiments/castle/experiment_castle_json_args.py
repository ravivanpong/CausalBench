import time
import logging
import os
import json
import argparse
from csv import DictWriter
import concurrent.futures
from castle.metrics import MetricsDAG
from causalbench.utils.helper import combine_multiple_lists

# use parameter file to set up experiment
def build_full_path(string):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.normpath(os.path.join(script_dir, string))


def init_func_with_param(func, kwargs: dict):
    if bool(kwargs):
        return func(**{k: v for k, v in kwargs.items() if v is not None})
    else:
        raise ValueError("kwargs can not be empty.")


def load_datasest(name: str, kwargs={}):
    if name.lower() == "alarm":
        from causalbench.data.alarm.alarm_loader import load_alarm

        return init_func_with_param(load_alarm, kwargs)
    elif name.lower() == "dream4":
        from causalbench.data.dream4.dream4_loader import load_dream4

        return init_func_with_param(load_dream4, kwargs)
    elif name.lower() == "jdk":
        from causalbench.data.jdk.jdk_loader import load_jdk

        return load_jdk()
    elif name.lower() == "postgres":
        from causalbench.data.postgres.postgres_loader import load_postgres

        return load_postgres()
    elif name.lower() == "sachs":
        from causalbench.data.sachs.sachs_loader import load_sachs

        return load_sachs()
    elif name.lower() == "networking":
        from causalbench.data.networking.networking_loader import load_networking

        return load_networking()
    elif name.lower() == "real_yacht":
        from causalbench.data.real_yacht.real_yacht_loader import load_real_yacht

        return load_real_yacht()
    elif name.lower() == "real_cites":
        from causalbench.data.real_cites.real_cites_loader import load_real_cites

        return load_real_cites()
    elif name.lower() == "real_auto_mpg":
        from causalbench.data.real_auto_mpg.real_auto_mpg_loader import (
            load_real_auto_mpg,
        )

        return load_real_auto_mpg()
    elif name.lower() == "simulated_feedback":
        from causalbench.data.simulated_feedback.simulated_feedback_loader import (
            load_feedback,
        )

        return load_feedback()
    else:
        raise ValueError(
            f"Data set: {name} not found. Please check info.txt for supported datasets."
        )


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


def gen_output_file(path_result, file_name, dict_result):
    """_summary_

    Args:
        path_result (_type_): _description_
        file_name (_type_): _description_
        dict_result (_type_): _description_
    """
    outfile = os.path.join(path_result, file_name)
    if os.path.exists(outfile):
        with open(outfile, "a", encoding="utf-8") as file:
            writer = DictWriter(file, dict_result.keys(), delimiter=";")
            writer.writerow(dict_result)
    else:
        with open(outfile, "w", encoding="utf-8") as file:
            writer = DictWriter(file, dict_result.keys(), delimiter=";")
            writer.writeheader()
            writer.writerow(dict_result)


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
    starttime = time.perf_counter()
    try:
        algo.learn(X)
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
        logging.info("%s on %s done.", algo_name, dataset_name)
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
            future = executor.submit(
                run,
                task[0]["name"],
                task[0]["kwargs"],
                task[1]["name"],
                task[1]["kwargs"],
                path_result,
                output_file_name,
            )
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CausalBench")
    parser.add_argument("-src", required=True, help="relative path to parameter file")
    param_file_path = build_full_path(parser.parse_args().src)
    logging.info("file to read: %s", param_file_path)

    params = None
    with open(param_file_path) as json_file:
        params = json.load(json_file)

    main()
