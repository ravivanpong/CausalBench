import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd


def load_hailf(index=1, sample_num=500, version=1):
    """_summary_
    Load HailFinder data set from local zip file.
    Default is version 1 of original network with 500 samples.

    Args:
        - index (int): number of titled networks. Accepted input are: [1, 3, 5, 10]
        - sample_num (int): number of samples. Accepted input are: [500, 1000, 5000]
        - version (int): version number. Accepted input are: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of HailFinder data set in form of Numpy NDArray
        - "X": HailFinder dataset in form of Numpy NDArray.
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
    hailf_target_bytes = None
    hailf_data_bytes = None
    zipfile_name = None
    if index == 1:
        zipfile_name = "hailf_data"
    elif index in [3, 5, 10]:
        zipfile_name = "hailf" + str(index) + "_data"
    else:
        raise ValueError(
            f"Index must be one of these values [1, 3, 5, 10]. Instead, {index} was given."
        )

    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/{zipfile_name}.zip") as zip_archive:
        if index == 1:
            hailf_target_bytes = zip_archive.read("HailFinder_graph.txt")
            hailf_data_bytes = zip_archive.read(
                f"HailFinder_s{sample_num}_v{version}.txt"
            )
        else:
            hailf_target_bytes = zip_archive.read(f"HailFinder{index}_graph.txt")
            hailf_data_bytes = zip_archive.read(
                f"HailFinder{index}_s{sample_num}_v{version}.txt"
            )
    # convert bytes into dadaframe
    true_graph_df = pd.read_fwf(BytesIO(hailf_target_bytes), header=None)
    data_df = pd.read_fwf(BytesIO(hailf_data_bytes), header=None)

    data = data_df.to_numpy()
    true_matrix = true_graph_df.to_numpy()

    result = {}
    result["true_matrix"] = true_matrix
    result["X"] = data
    result["var_num"] = data.shape[1]
    result["sample_num"] = data.shape[0]
    result["name"] = (
        f"HailFinder{index}_s{sample_num}_v{version}"
        if index != 1
        else f"HailFinder_s{sample_num}_v{version}"
    )
    return result


# hailf = load_hailf(3, 500, 5)
# print(hailf["var_num"])
# print(hailf["sample_num"])
# print(hailf["name"])
