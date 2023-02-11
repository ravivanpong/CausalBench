import os
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from causalbench.metrics.varsortability import varsortability


def load_alarm(index=1, sample_num=500, version=1):
    """_summary_
    Load alarm data set from local zip file.
    Default is version 1 of original network with 500 samples.

    Args:
        - index (int): number of titled networks. Accepted input are: [1, 3, 5, 10]
        - sample_num (int): number of samples. Accepted input are: [500, 1000, 5000]
        - version (int): version number. Accepted input are: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    Returns:
        result: dictionary with properties of:
        - "true_matrix": true graph of alarm data set in form of Numpy NDArray
        - "X": alarm dataset in form of Numpy NDArray. 11 variables x 7466 samples.
        - "var_num": number of variables
        - "sample_num": number of samples
        - "name": name of the dataset
        - "varsortability": measures how well the variance order reflects the causal order.
    """
    if sample_num not in [500, 1000, 5000]:
        raise ValueError("Sample number must be one of these values: [500, 1000, 5000]")
    if version not in list(range(1, 11)):
        raise ValueError(
            "Version must be one of these values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
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
        raise ValueError("Index must be one of these values [1, 3, 5, 10]")

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
    result["varsortability"] = varsortability(data, true_matrix)
    return result


# alarm = load_alarm(3, 1000, 5)
# print(alarm["var_num"])
# print(alarm["varsortability"])
# print(alarm["name"])
