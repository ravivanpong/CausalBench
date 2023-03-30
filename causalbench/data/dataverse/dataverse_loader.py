import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from causalbench.utils.helper import edges_to_matrix, dataframe_to_edges


def load_dataverse(with_hidden_var=True, is_big=False, max_parent_num=2, version=1):
    """_summary_
    Load dataverse data set from local zip file.

    Args:
        - with_hidden_var (bool): if True, use dataset with 3 hidden variable.
        - is_big (bool): if True, use dataset (no hidden variables) with more than or equal to 100 variables. Only effective when with_hidden_var is False.
        - max_parent_num (int): maximum parent num of each node. Only effective when is_big is False. Must be one of these values: [2, 3, 4, 5]
        - version (int): version number of dataset. Must be one of these values: [1, 2, 3, 4, 5].

    Returns:
        result: dictionary with properties of:
        - "with_hidden_var": if dataset with hidden variable
        - "true_matrix": true graph of dataverse data set in form of Numpy NDArray
        - "skeleton": skeleton of the graph (including spurious links due to common hidden cause) in form of Numpy NDArray. return None if no hidden variables.
        - "X": dataverse dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of data set
    """
    if not isinstance(with_hidden_var, bool):
        raise TypeError(
            f"with_hidden_var must be bool. Instead {type(with_hidden_var)} was given"
        )
    if not isinstance(is_big, bool):
        raise TypeError(f"is_big must be bool. Instead{type(is_big)} was given.")
    if max_parent_num not in [2, 3, 4, 5]:
        raise ValueError(
            f"max_parent_num must be one of these values: [2, 3, 4, 5]. Instead, {max_parent_num} was given."
        )
    if version not in [1, 2, 3, 4, 5]:
        raise ValueError(
            f"version must be one of these values [1, 2, 3, 4, 5]. Instead, {version} was given."
        )

    # read from zip file
    if with_hidden_var:
        dataverse_skeleton_bytes = None
    dataverse_target_bytes = None
    dataverse_data_bytes = None
    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/dataverse.zip") as zip_archive:
        if with_hidden_var:
            dataverse_target_bytes = zip_archive.read(
                f"graph/graph_with_hidden_variables/G{max_parent_num}_v{version}_confounders_target.csv"
            )
            dataverse_skeleton_bytes = zip_archive.read(
                f"graph/graph_with_hidden_variables/G{max_parent_num}_v{version}_confounders_skeleton.csv"
            )
            dataverse_data_bytes = zip_archive.read(
                f"graph/graph_with_hidden_variables/G{max_parent_num}_v{version}_confounders_numdata.csv"
            )
            dataset_name = f"G{max_parent_num}_v{version}_confounders_numdata.csv"
        else:
            if is_big:
                dataverse_target_bytes = zip_archive.read(
                    f"graph/graph_without_hidden_variables/Big_G3_v{version}_target.csv"
                )
                dataverse_data_bytes = zip_archive.read(
                    f"graph/graph_without_hidden_variables/Big_G3_v{version}_numdata.csv"
                )
                dataset_name = f"Big_G3_v{version}_numdata.csv"
            else:
                dataverse_target_bytes = zip_archive.read(
                    f"graph/graph_without_hidden_variables/G{max_parent_num}_v{version}_target.csv"
                )
                dataverse_data_bytes = zip_archive.read(
                    f"graph/graph_without_hidden_variables/G{max_parent_num}_v{version}_numdata.csv"
                )
                dataset_name = f"G{max_parent_num}_v{version}_numdata.csv"

    # convert bytes into dadaframe
    true_graph_df = pd.read_csv(BytesIO(dataverse_target_bytes))
    if with_hidden_var:
        skeleton_df = pd.read_csv(BytesIO(dataverse_skeleton_bytes))
    data_df = pd.read_csv(BytesIO(dataverse_data_bytes))

    # build true graph matrix
    nodes = list(data_df.columns)
    edges_true_graph = dataframe_to_edges(true_graph_df, "Cause", "Effect")
    true_matrix = edges_to_matrix(edges_true_graph, nodes)
    skeleton_matrix = None
    if with_hidden_var:
        edges_skeleton = dataframe_to_edges(skeleton_df, "Cause", "Effect")
        skeleton_matrix = edges_to_matrix(edges_skeleton, nodes)

    data = data_df.to_numpy()

    result = {}
    result["true_matrix"] = true_matrix
    result["with_hidden_var"] = with_hidden_var
    result["skeleton_matrix"] = skeleton_matrix
    result["X"] = data
    result["var_num"] = data.shape[1]
    result["sample_num"] = data.shape[0]
    result["name"] = dataset_name
    return result
