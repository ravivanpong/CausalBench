import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
import numpy as np

available_dataset_name = [
    "Network1_amp",
    "Network2_amp",
    "Network3_amp",
    "Network4_amp",
    "Network5_amp",
    "Network5_cont",
    "Network5_cont_p3n7",
    "Network5_cont_p7n3",
    "Network6_amp",
    "Network6_cont",
    "Network7_amp",
    "Network7_cont",
    "Network8_amp_amp",
    "Network8_amp_cont",
    "Network8_cont_amp",
    "Network9_amp_amp",
    "Network9_amp_cont",
    "Network9_cont_amp",
]


def load_feedback(name="Network1_amp", version=1):
    """_summary_
    Load feedback data set from local zip file.
    Default is version 1 of original network with 500 samples.

    Args:
        - name (string): Accepted input are:
            - Network1_amp
            - Network2_amp
            - Network3_amp
            - Network4_amp
            - Network5_amp
            - Network5_cont
            - Network5_cont_p3n7
            - Network5_cont_p7n3
            - Network6_amp
            - Network6_cont
            - Network7_amp
            - Network7_cont
            - Network8_amp_amp
            - Network8_amp_cont
            - Network8_cont_amp
            - Network9_amp_amp
            - Network9_amp_cont
            - Network9_cont_amp
        - version (int): version number. Accepted input are: [1, 2, 3, ... , 60]

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of feedback data set in form of Numpy NDArray
        - "X": feedback dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of the dataset
        - "varsortability": measures how well the variance order reflects the causal order.
    """
    if name not in available_dataset_name:
        raise ValueError(
            f"Sample number must be one of these values: {available_dataset_name}. Instead, {name} was given."
        )
    if version not in list(range(1, 61)):
        raise ValueError(
            f"Version must be one of these values: [1, 2, 3, ..., 60]. Instead, {version} was given."
        )
    # sim-25.Network6_amp.continuous.txt is empty. Problem of the source.
    if name == "Network6_amp" and version == 25:
        raise ValueError(
            f"sim-25.Network6_amp.continuous.txt is empty. Please choose another dataset."
        )
    if version in range(1, 10):
        version = "0" + str(version)

    # read from zip file
    feedback_target_bytes = None
    feedback_data_bytes = None

    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/feedback.zip") as zip_archive:
        feedback_target_bytes = zip_archive.read(f"{name}/{name}_target.csv")
        feedback_data_bytes = zip_archive.read(
            f"{name}/sim-{version}.{name}.continuous.txt"
        )
    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(feedback_target_bytes), sep=";")
    data_df = pd.read_csv(BytesIO(feedback_data_bytes), sep="\t")

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
    result["name"] = f"sim-{version}.{name}.continuous"
    return result


# feedback = load_feedback(name="Network5_amp", version=3)
# print(feedback["var_num"])
# print(feedback["sample_num"])
# print(feedback["name"])
