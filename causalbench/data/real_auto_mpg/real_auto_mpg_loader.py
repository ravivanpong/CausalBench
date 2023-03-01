import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np


def load_real_auto_mpg():
    """Load real_auto_mpg data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of real_auto_mpg data set in form of Numpy NDArray
        - "X": real_auto_mpg dataset in form of Numpy NDArray. 392 samples and 8 variables.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
    """
    # read from zip file
    real_auto_mpg_target_bytes = None
    real_auto_mpg_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/real_auto_mpg.zip") as zip_archive:
        real_auto_mpg_target_bytes = zip_archive.read("real_auto_mpg_target.csv")
        real_auto_mpg_data_bytes = zip_archive.read("real_auto_mpg_data.txt")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(real_auto_mpg_target_bytes), sep=";")
    data_df = pd.read_csv(BytesIO(real_auto_mpg_data_bytes), sep="\t")

    # build true graph matrix
    nodes = list(data_df.columns)
    edges = []
    for _, row in true_graph_df.iterrows():
        edges.append((row["source"], row["target"]))

    node_index_map = {}
    i = 0
    for node in nodes:
        node_index_map[node] = i
        i += 1

    n_nodes = len(nodes)

    true_matrix = np.zeros((n_nodes, n_nodes))
    for edge in edges:
        true_matrix[node_index_map[edge[0]]][node_index_map[edge[1]]] = 1

    data = data_df.to_numpy()

    result = {}
    result["true_matrix"] = true_matrix
    result["X"] = data
    result["var_num"] = data.shape[1]
    result["sample_num"] = data.shape[0]
    result["name"] = "real_auto_mpg"
    return result


# real_auto_mpg = load_real_auto_mpg()
# print(real_auto_mpg["var_num"])
# print(real_auto_mpg["sample_num"])
