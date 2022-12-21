import time
import logging
import os
from csv import DictWriter
import numpy as np
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
import concurrent.futures

# all = [
#     "ICALiNGAM",
#     "DirectLiNGAM",
#     "BootstrapResult",
#     "MultiGroupDirectLiNGAM",
#     "CausalEffect",
#     "VARLiNGAM",
#     "VARMALiNGAM",
#     "LongitudinalLiNGAM",
#     "VARBootstrapResult",
#     "VARMABootstrapResult",
#     "LongitudinalBootstrapResult",
#     "BottomUpParceLiNGAM",
#     "RCD",
#     "CAMUV",
#     "LiNA",
#     "MDLiNA",
#     "RESIT",
#     "LiM"
# ]


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
    if algo_name.lower() == "directlingam":
        from lingam import DirectLiNGAM

        algo = DirectLiNGAM()
    elif algo_name.lower() == "icalingam":
        from lingam import ICALiNGAM

        algo = ICALiNGAM()
    elif algo_name.lower() == "multigroupdirectlingam":
        from lingam import MultiGroupDirectLiNGAM

        algo = MultiGroupDirectLiNGAM()
    elif algo_name.lower() == "varlingam":
        from lingam import VARLiNGAM

        algo = VARLiNGAM()
    elif algo_name.lower() == "varmalingam":
        from lingam import VARMALiNGAM

        algo = VARMALiNGAM()
    elif algo_name.lower() == "longitudinallingam":
        from lingam import LongitudinalLiNGAM

        algo = LongitudinalLiNGAM()
    elif algo_name.lower() == "lim":
        from lingam import LiM  # pairwise

        algo = LiM()
    # bottomupparcelingam, CAM-UV, rcd, lina, mdlina, resit are taged as pairwise
    else:
        raise ValueError("Unknown algorithm.==========")

    logging.info(f"import algorithm corresponding to {algo_name} complete!")

    # structure learning
    starttime = time.perf_counter()
    algo.fit(X)
    finishtime = time.perf_counter()
    runtime = round(finishtime - starttime, 2)
    # estimated_causal_matrix = (abs(algo.adjacency_matrix_) > 0).astype(int).T # nan to 0
    estimated_causal_matrix = np.where(algo.adjacency_matrix_ == 0, 0, 1)  # nan to 1

    # covert weighted adj_matrix to causal_matrix

    # calculate metrics
    mt = MetricsDAG(estimated_causal_matrix, true_causal_matrix)

    dict_result = {
        "dataset_name": "random simulated",
        "N_variables": X.shape[1],
        "N_obs": X.shape[0],
        "model_name": algo_name.lower(),
        "library_name": "lingam",
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
    gen_output_file(path_result, "lingam_example_result.csv", dict_result)


def main():
    # algorithms to run (cpu only)
    algo_name_list = ["icalingam", "DirectLiNGAM", "multigroupdirectlingam"]
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
