import time
import logging
import os
from csv import DictWriter
from castle.metrics import MetricsDAG
import concurrent.futures
from causalbench.utils.helper import combine_multiple_lists

###################### Edit blow area to set up experiment. ##############################
# step 1: set dataset and it's parameters
dataset_list = {
    "alarm": {"index": None, "sample_num": None, "version": None},
    # "dream4": {"version": None},
    # "jdk": {},
    # "postgres": {},
    "sachs": {},
}
# step 2:  set algorithm and it's parameters
algo_param_dict = {
    "pc": {"variant": None, "ci_test": None, "alpha": None, "priori_knowledge": None},
    "ges": {"criterion": None, "method": None, "k": None, "N": None},
}
# step 3: set output file name
OUTPUT_FILE_NAME = "gcastle_experiment"
######################  Edit above area to set up experiment. #############################


def load_datasest(dataset_name: str):
    if dataset_name.lower() == "alarm":
        from causalbench.data.alarm.alarm_loader import load_alarm

        kwargs = dataset_list["alarm"]

        return load_alarm(**{k: v for k, v in kwargs.items() if v is not None})
    elif dataset_name.lower() == "dream4":
        from causalbench.data.dream4.dream4_loader import load_dream4

        kwargs = dataset_list["dream4"]

        return load_dream4(**{k: v for k, v in kwargs.items() if v is not None})
    elif dataset_name.lower() == "jdk":
        from causalbench.data.jdk.jdk_loader import load_jdk

        return load_jdk()
    elif dataset_name.lower() == "postgres":
        from causalbench.data.postgres.postgres_loader import load_postgres

        return load_postgres()
    elif dataset_name.lower() == "sachs":
        from causalbench.data.sachs.sachs_loader import load_sachs

        return load_sachs()
    else:
        raise ValueError(f"Data set: {dataset_name} not found.")


def init_algo_from_gcastle(algo_name: str):
    """_summary_

    Args:
        algo_name (str): _description_

    Raises:
        ValueError: _description_ß

    Returns:
        _type_: _description_
    """
    if algo_name.lower() == "pc":
        from castle.algorithms import PC

        kwargs = algo_param_dict["pc"]

        return PC(**{k: v for k, v in kwargs.items() if v is not None})
    elif algo_name.lower() == "ges":
        from castle.algorithms import GES

        kwargs = algo_param_dict["ges"]

        return GES(**{k: v for k, v in kwargs.items() if v is not None})
    # elif algo_name.lower() == "icalingam":
    #     from castle.algorithms import ICALiNGAM

    #     return ICALiNGAM()
    # elif algo_name.lower() == "directlingam":
    #     from castle.algorithms import DirectLiNGAM

    #     return DirectLiNGAM()
    # elif algo_name.lower() == "anm": #pairwise
    #     from castle.algorithms import ANMNonlinear
    #     return ANMNonlinear()
    # These algorithms need GPU
    # elif algo.lower() == "golem":
    #     from castle.algorithms import GOLEM
    #     return GOLEM()
    # elif algo.lower() == "grandag":
    #     from castle.algorithms import GraNDAG
    #     return GraNDAG()
    # elif algo_name.lower() == "notears":
    #     from castle.algorithms import Notears
    #     return Notears()
    # elif algo_name.lower() == "notearslowrank":
    #     from castle.algorithms import NotearsLowRank
    #     return NotearsLowRank()
    # elif algo_name.lower() == "notearsnonlinear": #unclear
    #     from castle.algorithms import NotearsNonlinear
    #     return NotearsNonlinear()
    # elif algo_name.lower() == "corl":
    #     from castle.algorithms import CORL
    #     return CORL()
    # elif algo_name.lower() == "rl":
    #     from castle.algorithms import RL
    #     return RL()
    # elif algo.lower() == "gae":
    #     from castle.algorithms import GAE
    #     return GAE()
    # elif algo.lower() == "pnl": # pairwise?
    #     from castle.algorithms import PNL
    #     return PNL()
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
    algo = init_algo_from_gcastle(algo_name)
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
    dataset_name_list = list(dataset_list.keys())
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


if __name__ == "__main__":
    main()
