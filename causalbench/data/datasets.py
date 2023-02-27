##
## Author: Ployplearn Ravivanpong
##
##
##
##
##
##
##
##


import os
import pandas as pd
import numpy as np
import time
from causalbench.utils.helper import load_datasest


def list_datasets(self):
    """
    List all the datasets avaialble in the packages
    """


def summarize(dataset="all", file=None, update=None):
    """
    Give an overview summary of a given or all the available datasets

    Returns
    --------
    data_summary : pd.DataFrame : contains
        [dataset name (variant name), number of variables, number of samples, varsortability]
    """

    ## If a output file path is given, then use it


def update_dataset_info(dataset_name: str, kwargs: dict, summary_df: pd.DataFrame):
    dataset = load_datasest(dataset_name, kwargs)
    name, var_num, sample_num, varsortability = (
        dataset["name"],
        dataset["var_num"],
        dataset["sample_num"],
        dataset["varsortability"],
    )
    print(f"{name} loaded")

    if name in summary_df["name"].values:
        row_index = np.where(summary_df["name"] == name)
        print(row_index[0])
        print(df.loc[row_index[0][0], "name"])
        df.loc[row_index[0][0], "var_num"] = var_num
        df.loc[row_index[0][0], "sample_num"] = sample_num
        df.loc[row_index[0][0], "varsortability"] = varsortability
        df.loc[row_index[0][0], "last_updated"] = time.ctime()
        return summary_df

    else:
        print(f"{name} not found, about to append")
        row = {
            "name": name,
            "var_num": var_num,
            "sample_num": sample_num,
            "varsortability": varsortability,
            "last_updated": time.ctime(),
        }
        return pd.concat(
            [summary_df, pd.DataFrame([row])],
            axis=0,
            ignore_index=True,
        )


if __name__ == "__main__":
    dirname = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(f"{dirname}/datasets_summary.csv")

    # df = update_dataset_info("jdk", {}, df)

    # versions = [1, 2, 3, 4]
    # for version in versions:
    #     kwargs = {"version": version}
    #     df = update_dataset_info("dream4", kwargs, df)

    sample_nums = [500, 1000, 5000]
    versions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for sample_num in sample_nums:
        for version in versions:
            kwargs = {"sample_num": sample_num, "version": version}
            df = update_dataset_info("mildew", kwargs, df)

    # indexs = [1, 3, 5, 10]
    # sample_nums = [500, 1000, 5000]
    # versions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for index in indexs:
    #     for sample_num in sample_nums:
    #         for version in versions:
    #             kwargs = {"index": index, "sample_num": sample_num, "version": version}
    #             df = update_dataset_info("insurance", kwargs, df)

    df.to_csv(f"{dirname}/datasets_summary.csv", index=False)
