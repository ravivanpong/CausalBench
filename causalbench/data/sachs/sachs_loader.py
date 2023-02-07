import zipfile
import pandas as pd
from causalbench.metrics.varsortability import varsortability

# unzip
def unzip(from_path, to_dir):
    with zipfile.ZipFile(
        from_path,
        "r",
    ) as zip_ref:
        zip_ref.extractall(to_dir)


def sachs_loader(path_to_dataset):
    true_causal_matrix_df = pd.read_fwf(
        "/Users/huigong/ba-thesis/CausalBench/causalbench/data/sachs/ground_truth.txt",
        header=None,
    )
    true_causal_matrix = true_causal_matrix_df.to_numpy()
    X_df = pd.read_fwf(path_to_dataset, skiprows=1, header=None)
    X = X_df.to_numpy()
    result = {}
    result["true_causal_matrix"] = true_causal_matrix
    result["X"] = X
    result["var_num"] = X.shape[1]
    result["sample_num"] = X.shape[0]
    result["varsortability"] = varsortability(X, true_causal_matrix)
    print(result["varsortability"])
    return result


# unzip(
#     "/Users/huigong/ba-thesis/CausalBench/causalbench/data/sachs/real-sachs.zip",
#     "/Users/huigong/ba-thesis/CausalBench/causalbench/data/sachs",
# )

sachs = sachs_loader(
    "/Users/huigong/ba-thesis/CausalBench/causalbench/data/sachs/data/sachs.2005.discrete.txt"
)
