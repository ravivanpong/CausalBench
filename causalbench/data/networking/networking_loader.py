import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np
from causalbench.metrics.varsortability import varsortability


def load_networking():
    """Load networking data set from local zip file.

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of networking data set in form of Numpy NDArray
        - "X": networking dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
        - "varsortability": measures how well the variance order reflects the causal order.
    """
    # read from zip file
    networking_target_bytes = None
    networking_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/networking_data.zip") as zip_archive:
        networking_target_bytes = zip_archive.read("networking_target.csv")
        networking_data_bytes = zip_archive.read("networking_data.csv")
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(networking_target_bytes), sep=";")
    data_df = pd.read_csv(BytesIO(networking_data_bytes))
    data_df = data_df.iloc[:, 1:18]  # slice out first column which is unnamed
    data_df = data_df.drop(
        columns=["site", "url", "content_type", "server"]
    )  # drop string valued and not used variable
    dummy_time = pd.get_dummies(data_df["time"])
    dummy_server_class = pd.get_dummies(data_df["server.class"])
    data_df = pd.concat([data_df, dummy_time, dummy_server_class], axis=1)

    data_df = data_df.drop(columns=["time", "server.class"])
    data_df.replace({False: 0, True: 1}, inplace=True)

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
    result["name"] = "networking"
    result["varsortability"] = varsortability(data, true_matrix)
    return result


# networking = load_networking()
# print(networking["var_num"])
# print(networking["varsortability"])
# print(networking["sample_num"])
