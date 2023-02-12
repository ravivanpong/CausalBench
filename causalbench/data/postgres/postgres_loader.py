import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np
from causalbench.metrics.varsortability import varsortability


def load_postgres():
    """Load postgres data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of postgres data set in form of Numpy NDArray
        - "X": postgres dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
        - "varsortability": measures how well the variance order reflects the causal order.
    """
    # read from zip file
    postgres_target_bytes = None
    postgres_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/postgres_data.zip") as zip_archive:
        postgres_target_bytes = zip_archive.read("postgres_target.csv")
        postgres_data_bytes = zip_archive.read("postgres_data.csv")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(postgres_target_bytes), sep=";")
    data_df = pd.read_csv(BytesIO(postgres_data_bytes))
    data_df = data_df.iloc[:, 1:23]  # slice out first column which is url string

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
    result["name"] = "postgres"
    result["varsortability"] = varsortability(data, true_matrix)
    return result


postgres = load_postgres()
print(postgres["var_num"])
print(postgres["varsortability"])
print(postgres["sample_num"])
