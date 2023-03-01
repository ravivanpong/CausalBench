import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd


def load_link(sample_num=500, version=1):
    """_summary_
    Load link dataset from local zip file.
    Default is version 1 of original network with 500 samples.

    Args:
        - sample_num (int): number of samples. Accepted input are: [500, 1000, 5000]
        - version (int): version number. Accepted input are: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of link data set in form of Numpy NDArray
        - "X": link dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of the dataset
    """
    if sample_num not in [500, 1000, 5000]:
        raise ValueError(
            f"Sample number must be one of these values: [500, 1000, 5000]. Instead, {sample_num} was given."
        )
    if version not in list(range(1, 11)):
        raise ValueError(
            f"Version must be one of these values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Instead, {version} was given."
        )

    # read from zip file
    link_target_bytes = None
    link_data_bytes = None
    zipfile_name = "link_data"

    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/{zipfile_name}.zip") as zip_archive:
        link_target_bytes = zip_archive.read("Link_graph.txt")
        link_data_bytes = zip_archive.read(f"Link_s{sample_num}_v{version}.txt")
    # convert bytes into dadaframe
    true_graph_df = pd.read_fwf(BytesIO(link_target_bytes), header=None)
    data_df = pd.read_csv(BytesIO(link_data_bytes), header=None, sep="\s+")

    data = data_df.to_numpy()
    true_matrix = true_graph_df.to_numpy()

    result = {}
    result["true_matrix"] = true_matrix
    result["X"] = data
    result["var_num"] = data.shape[1]
    result["sample_num"] = data.shape[0]
    result["name"] = f"link_s{sample_num}_v{version}"
    return result


# link = load_link(1000, 1)
# print(link["var_num"])
# print(link["sample_num"])
# print(link["name"])
