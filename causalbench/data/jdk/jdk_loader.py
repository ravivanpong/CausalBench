import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np


def load_jdk():
    """Load jdk data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of jdk data set in form of Numpy NDArray
        - "X": jdk dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
    """
    # read from zip file
    jdk_target_bytes = None
    jdk_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/jdk_data.zip") as zip_archive:
        jdk_target_bytes = zip_archive.read("jdk_target.csv")
        jdk_data_bytes = zip_archive.read("jdk_data.csv")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(jdk_target_bytes), sep=";")
    data_df = pd.read_csv(BytesIO(jdk_data_bytes))
    data_df = data_df.iloc[
        :, 1:16
    ]  # slice out first column which is string of repo name

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
    result["name"] = "jdk"
    return result
