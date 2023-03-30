import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np


def load_dream4(version=1):
    """Load dream4 data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of dream4 data set in form of Numpy NDArray
        - "X": dream4 dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
    """
    if version not in [1, 2, 3, 4]:
        raise ValueError("Version of dream4 must be one of these values: [1, 2, 3, 4]")
    # read from zip file
    dream4_target_bytes = None
    dream4_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/dream4_{version}.zip") as zip_archive:
        dream4_target_bytes = zip_archive.read(f"dream4_{version}_target.csv")
        dream4_data_bytes = zip_archive.read(f"dream4_{version}_data.csv")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(dream4_target_bytes), header=None, sep=";")
    data_df = pd.read_csv(BytesIO(dream4_data_bytes), sep=";")
    data_df = data_df.iloc[:, 1:101]  # slice out first column which is unnamed

    # build true graph matrix
    nodes = list(data_df.columns)
    edges = []
    for _, row in true_graph_df.iterrows():
        edges.append((row[0], row[1]))

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
    result["name"] = f"dream4_{version}"
    return result
