import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np
from causalbench.metrics.varsortability import varsortability


def load_sachs():
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
    result["varsortability"] = varsortability(data, true_matrix)
    return result


# sachs = load_sachs()
# print(sachs["var_num"])
# print(sachs["varsortability"])
# print(sachs["sample_num"])
