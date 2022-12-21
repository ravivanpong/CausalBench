import time
import logging
import os
from csv import DictWriter
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
import concurrent.futures


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


def load_data():
    # data simulation, simulate true causal dag and train_data.
    weighted_random_dag = DAG.erdos_renyi(
        n_nodes=10, n_edges=10, weight_range=(0.5, 2.0), seed=1
    )
    dataset = IIDSimulation(
        W=weighted_random_dag, n=1000, method="linear", sem_type="gauss"
    )
    true_causal_matrix, X = dataset.B, dataset.X
    print(f"type of true_causal_matrix, X: {type(true_causal_matrix), type(X)}")
    return true_causal_matrix, X


def run(true_causal_matrix, X, algo_name: str, path_result):
    if algo_name.lower() == "pc":
        from castle.algorithms import PC

        algo = PC()
    elif algo_name.lower() == "icalingam":
        from castle.algorithms import ICALiNGAM

        algo = ICALiNGAM()
    elif algo_name.lower() == "directlingam":
        from castle.algorithms import DirectLiNGAM

        algo = DirectLiNGAM()

    # elif algo_name.lower() == "ges":
    #     from castle.algorithms import GES

    #     algo = GES()
    # elif algo_name.lower() == "anm": #pairwise
    #     from castle.algorithms import ANMNonlinear
    #     algo = ANMNonlinear()
    # These algorithms need GPU
    # elif algo.lower() == "golem":
    #     from castle.algorithms import GOLEM
    #     algo = GOLEM()
    # elif algo.lower() == "grandag":
    #     from castle.algorithms import GraNDAG
    #     algo = GraNDAG()
    # elif algo_name.lower() == "notears":
    #     from castle.algorithms import Notears
    #     algo = Notears()
    # elif algo_name.lower() == "notearslowrank":
    #     from castle.algorithms import NotearsLowRank
    #     algo = NotearsLowRank()
    # elif algo_name.lower() == "notearsnonlinear": #unclear
    #     from castle.algorithms import NotearsNonlinear
    #     algo = NotearsNonlinear()
    # elif algo_name.lower() == "corl":
    #     from castle.algorithms import CORL
    #     algo = CORL()
    # elif algo_name.lower() == "rl":
    #     from castle.algorithms import RL
    #     algo = RL()
    # elif algo.lower() == "gae":
    #     from castle.algorithms import GAE
    #     algo = GAE()
    # elif algo.lower() == "pnl": # pairwise?
    #     from castle.algorithms import PNL
    #     algo = PNL()
    else:
        raise ValueError("Unknown algorithm.==========")

    logging.info(f"import algorithm corresponding to {algo_name} complete!")

    # structure learning
    starttime = time.perf_counter()
    algo.learn(X)
    finishtime = time.perf_counter()
    runtime = round(finishtime - starttime, 2)

    # plot predict_dag and true_dag
    # GraphDAG(pc.causal_matrix, true_causal_matrix, "result")

    # calculate metrics
    mt = MetricsDAG(algo.causal_matrix, true_causal_matrix)

    dict_result = {
        "dataset_name": "random simulated",
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
    gen_output_file(path_result, "gcastle_example_result.csv", dict_result)


def main():
    # algorithms to run (cpu only)
    algo_name_list = ["pc", "icalingam", "DirectLiNGAM"]
    # load data
    true_causal_matrix, X = load_data()
    # set output file path
    file_dir = os.path.dirname(__file__)
    path_result = os.path.join(file_dir, "result")
    try:
        os.mkdir(path_result)
    except FileExistsError:
        logging.info("result dir already exists")
    else:
        logging.info("result dir created.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for algo_name in algo_name_list:
            executor.submit(run, true_causal_matrix, X, algo_name, path_result)


if __name__ == "__main__":
    main()
