import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd


def load_alarm(index=1, sample_num=500, version=1):
    """_summary_
    Load alarm data set from local zip file.
    Default is version 1 of original network with 500 samples.

    Args:
        - index (int): number original networks that are tiled together to get a larger network. Accepted input are: [1, 3, 5, 10]
        - sample_num (int): number of samples. Accepted input are: [500, 1000, 5000]
        - version (int): version number. Accepted input are: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of alarm data set in form of Numpy NDArray
        - "X": alarm dataset in form of Numpy NDArray.
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
    alarm_target_bytes = None
    alarm_data_bytes = None
    zipfile_name = None
    if index == 1:
        zipfile_name = "alarm_data"
    elif index in [3, 5, 10]:
        zipfile_name = "alarm" + str(index) + "_data"
    else:
        raise ValueError(
            f"Index must be one of these values [1, 3, 5, 10]. Instead, {index} was given."
        )

    dirname = os.path.dirname(os.path.realpath(__file__))
    with ZipFile(f"{dirname}/{zipfile_name}.zip") as zip_archive:
        alarm_target_bytes = zip_archive.read(f"Alarm{index}_graph.txt")
        alarm_data_bytes = zip_archive.read(
            f"Alarm{index}_s{sample_num}_v{version}.txt"
        )
    # convert bytes into dadaframe
    true_graph_df = pd.read_fwf(BytesIO(alarm_target_bytes), header=None)
    data_df = pd.read_fwf(BytesIO(alarm_data_bytes), header=None)

    data = data_df.to_numpy()
    true_matrix = true_graph_df.to_numpy()

    result = {}
    result["true_matrix"] = true_matrix
    result["X"] = data
    result["var_num"] = data.shape[1]
    result["sample_num"] = data.shape[0]
    result["name"] = f"Alarm{index}_s{sample_num}_v{version}"
    return result
