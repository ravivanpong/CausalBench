import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np
from causalbench.metrics.varsortability import varsortability


def load_real_cites():
    """Load real_cites data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of real_cites data set in form of Numpy NDArray
        - "X": real_cites dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
        - "varsortability": measures how well the variance order reflects the causal order.
    """
    # read from zip file
    real_cites_target_bytes = None
    real_cites_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/real_cites.zip") as zip_archive:
        real_cites_target_bytes = zip_archive.read("cites_target.csv")
        real_cites_data_bytes = zip_archive.read("cites_data.txt")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(real_cites_target_bytes), sep=";")
    data_df = pd.read_csv(BytesIO(real_cites_data_bytes), sep=" ")
    print(true_graph_df)
    print(data_df)

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
    result["name"] = "real_cites"
    result["varsortability"] = varsortability(data, true_matrix)
    return result


# real_cites = load_real_cites()
# print(real_cites["var_num"])
# print(real_cites["varsortability"])
# print(real_cites["sample_num"])
