import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from causalbench.metrics.varsortability import varsortability


def load_munin1(sample_num=500, version=1):
    """_summary_
    Load munin1 dataset from local zip file.
    Default is version 1 of original network with 500 samples.

    Args:
        - sample_num (int): number of samples. Accepted input are: [500, 1000, 5000]
        - version (int): version number. Accepted input are: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of munin1 data set in form of Numpy NDArray
        - "X": munin1 dataset in form of Numpy NDArray.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of the dataset
        - "varsortability": measures how well the variance order reflects the causal order.
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
    munin1_target_bytes = None
    munin1_data_bytes = None
    zipfile_name = "munin_data"

    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/{zipfile_name}.zip") as zip_archive:
        munin1_target_bytes = zip_archive.read("Munin1_graph.txt")
        munin1_data_bytes = zip_archive.read(f"Munin1_s{sample_num}_v{version}.txt")
    # convert bytes into dadaframe
    true_graph_df = pd.read_fwf(BytesIO(munin1_target_bytes), header=None)
    data_df = pd.read_csv(BytesIO(munin1_data_bytes), header=None, sep="\s+")

    data = data_df.to_numpy()
    true_matrix = true_graph_df.to_numpy()

    result = {}
    result["true_matrix"] = true_matrix
    result["X"] = data
    result["var_num"] = data.shape[1]
    result["sample_num"] = data.shape[0]
    result["name"] = f"munin1_s{sample_num}_v{version}"
    result["varsortability"] = varsortability(data, true_matrix)
    return result


# munin1 = load_munin1(500, 10)
# print(munin1["var_num"])
# print(munin1["varsortability"])
# print(munin1["name"])
