import time
import logging
import os
from csv import DictWriter
import concurrent.futures
from castle.metrics import MetricsDAG
from causalbench.utils.helper import combine_multiple_lists

###################### Edit blow area to set up experiment. ##############################
# step 1: set dataset and it's parameters
dataset_list = {
    # "alarm": {"index": 3},
    # "dream4": {"version": None},
    # "jdk": {},
    # "postgres": {},
    # "sachs": {},
    # "networking": {},
    # "real_auto_mpg": {},
    # "real_cites": {},
    "real_yacht": {},
    # "simulated_feedback": {}
}
# step 2:  set algorithm and it's parameters
algo_param_dict = {
    # "pc": {"variant": None, "ci_test": None, "alpha": None, "priori_knowledge": None},
    # "ges": {"criterion": None, "method": None, "k": None, "N": None},
    # "icalingam": {},
    # "directlingam": {},
    # "anm": {},
    # "golem": {},
    # "grandag": {},
    # "notears": {},
    # "notearslowrank": {},
    # "notearsnonlinear": {},
    "corl": {},
    # "rl": {},
    # "gae": {},
    # "pnl": {},
}
# step 3: set output file name
OUTPUT_FILE_NAME = "yacht_castle"
######################  Edit above area to set up experiment. #############################


def init_func_with_param(func, kwargs):
    if bool(kwargs):
        return func(**{k: v for k, v in kwargs.items() if v is not None})
    return func()


def load_datasest(dataset_name: str):
    kwargs = dataset_list[dataset_name]
    if dataset_name.lower().startswith("alarm"):
        from causalbench.data.alarm.alarm_loader import load_alarm

        return init_func_with_param(load_alarm, kwargs)
    elif dataset_name.lower().startswith("dream4"):
        from causalbench.data.dream4.dream4_loader import load_dream4

        return init_func_with_param(load_dream4, kwargs)
    elif "jdk" == dataset_name.lower():
        from causalbench.data.jdk.jdk_loader import load_jdk

        return load_jdk()
    elif "postgres" == dataset_name.lower():
        from causalbench.data.postgres.postgres_loader import load_postgres

        return load_postgres()
    elif "sachs" == dataset_name.lower():
        from causalbench.data.sachs.sachs_loader import load_sachs

        return load_sachs()
    elif "networking" == dataset_name.lower():
        from causalbench.data.networking.networking_loader import load_networking

        return load_networking()
    elif "real_yacht" == dataset_name.lower():
        from causalbench.data.real_yacht.real_yacht_loader import load_real_yacht

        return load_real_yacht()
    elif "real_cities" == dataset_name.lower():
        from causalbench.data.real_cites.real_cites_loader import load_real_cites

        return load_real_cites()
    elif "real_auto_mpg" == dataset_name.lower():
        from causalbench.data.real_auto_mpg.real_auto_mpg_loader import (
            load_real_auto_mpg,
        )

        return load_real_auto_mpg()
    elif "simulated_feedback" == dataset_name.lower():
        from causalbench.data.simulated_feedback.simulated_feedback_loader import (
            load_feedback,
        )

        return load_feedback()
    else:
        raise ValueError(f"Data set: {dataset_name} not found.")


def init_algo_from_gcastle(algo_name: str, var_num: int):
    """_summary_

    Args:
        algo_name (str): _description_

    Raises:
        ValueError: _description_ß

    Returns:
        _type_: _description_
    """
    kwargs = algo_param_dict[algo_name]
    if algo_name.lower().startswith("pc"):
        from castle.algorithms import PC

        return init_func_with_param(PC, kwargs)
    elif algo_name.lower().startswith("ges"):
        from castle.algorithms import GES

        return init_func_with_param(GES, kwargs)
    elif algo_name.lower().startswith("icalingam"):
        from castle.algorithms import ICALiNGAM

        return init_func_with_param(ICALiNGAM, kwargs)
    elif algo_name.lower().startswith("directlingam"):
        from castle.algorithms import DirectLiNGAM

        return init_func_with_param(DirectLiNGAM, kwargs)
    elif algo_name.lower().startswith("anm"):  # pairwise
        from castle.algorithms import ANMNonlinear

        return init_func_with_param(ANMNonlinear, kwargs)
    # These algorithms need GPU
    elif algo_name.lower().startswith("golem"):
        from castle.algorithms import GOLEM

        return init_func_with_param(GOLEM, kwargs)
    elif algo_name.lower().startswith("grandag"):
        from castle.algorithms import GraNDAG

        kwargs["input_dim"] = var_num
        return init_func_with_param(GraNDAG, kwargs)
    elif algo_name.lower().startswith("notears"):
        from castle.algorithms import Notears

        return init_func_with_param(Notears, kwargs)
    elif algo_name.lower().startswith("notearslowrank"):
        from castle.algorithms import NotearsLowRank

        return init_func_with_param(NotearsLowRank, kwargs)
    elif algo_name.lower().startswith("notearsnonlinear"):  # unclear
        from castle.algorithms import NotearsNonlinear

        return init_func_with_param(NotearsNonlinear, kwargs)
    elif algo_name.lower().startswith("corl"):
        from castle.algorithms import CORL

        return init_func_with_param(CORL, kwargs)
    elif algo_name.lower().startswith("rl"):
        from castle.algorithms import RL

        return init_func_with_param(RL, kwargs)
    elif algo_name.lower().startswith("gae"):
        from castle.algorithms import GAE

        return GAE()
    elif algo_name.lower().startswith("pnl"):  # pairwise?
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


def run(dataset_name: str, algo_name: str, path_result):
    # load data set
    dataset = load_datasest(dataset_name)
    true_causal_matrix = dataset["true_matrix"]
    X = dataset["X"]
    logging.info("%s loaded", dataset_name)
    # init algorithm
    var_num = dataset["var_num"]
    algo = init_algo_from_gcastle(algo_name, var_num)
    logging.info("%s algorithm initiated.", algo_name)
    # structure learning
    starttime = time.perf_counter()
    algo.learn(X)
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
        "model_name": algo_name.lower(),
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
    gen_output_file(path_result, f"{OUTPUT_FILE_NAME}.csv", dict_result)


def main():
    algo_name_list = list(algo_param_dict.keys())
    logging.info(f"algorithms are: {algo_name_list}")
    dataset_name_list = list(dataset_list.keys())
    logging.info(f"data set are: {dataset_name_list}")
    # set output file path
    file_dir = os.path.dirname(__file__)
    path_result = os.path.join(file_dir, "result")
    try:
        os.mkdir(path_result)
    except FileExistsError:
        logging.info("result dir already exists")
    else:
        logging.info("result dir created.")

    task_list = combine_multiple_lists([dataset_name_list, algo_name_list])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for task in task_list:
            executor.submit(run, task[0], task[1], path_result)
    # for task in task_list:
    #     run(task[0], task[1], path_result)


if __name__ == "__main__":
    main()
