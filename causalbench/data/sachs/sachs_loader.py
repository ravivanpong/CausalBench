import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np


def load_sachs():
    """Load sachs data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of sachs data set in form of Numpy NDArray
        - "X": sachs dataset in form of Numpy NDArray. 11 variables x 7466 samples.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
    """
    # read from zip file
    sachs_target_bytes = None
    sachs_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/sachs.zip") as zip_archive:
        sachs_target_bytes = zip_archive.read("sachs/cyto_full_target.csv")
        sachs_data_bytes = zip_archive.read("sachs/cyto_full_data.csv")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(sachs_target_bytes))
    data_df = pd.read_csv(BytesIO(sachs_data_bytes))
    # build true graph matrix
    nodes = list(data_df.columns)
    edges = []
    for _, row in true_graph_df.iterrows():
        edges.append((row["Cause"], row["Effect"]))

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
    result["name"] = "sachs"
    return result


# sachs = load_sachs()
# print(sachs["var_num"])
# print(sachs["sample_num"])
