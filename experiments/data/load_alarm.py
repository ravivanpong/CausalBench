import zipfile
import pandas as pd

# unzip
def unzip(from_path, to_dir):
    with zipfile.ZipFile(
        "/Users/huigong/ba-thesis/CausalBench/causalbench/data/alarm/alarm_data.zip",
        "r",
    ) as zip_ref:
        zip_ref.extractall(
            "/Users/huigong/ba-thesis/CausalBench/experiments/data/alarm_data"
        )


# convert txt file to pd dataframe
# /Users/huigong/ba-thesis/CausalBench/experiments/data/alarm_data/Alarm1_s500_v1.txt
def alarm_loader(path_to_dataset):
    true_causal_matrix = pd.read_fwf(
        "/Users/huigong/ba-thesis/CausalBench/experiments/data/alarm_data/Alarm1_graph.txt",
        header=None,
    )
    X = pd.read_fwf(path_to_dataset, header=None)
    return true_causal_matrix, X


true_causal_matrix, X = alarm_loader(
    "/Users/huigong/ba-thesis/CausalBench/experiments/data/alarm_data/Alarm1_s500_v1.txt"
)
print(true_causal_matrix.shape)
print(X.shape)
