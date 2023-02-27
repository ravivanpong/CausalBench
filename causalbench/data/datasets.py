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

    # with_hidden_var_bools = [True, False]
    # is_big_bools = [True, False]
    # max_parent_nums = [2, 3, 4, 5]
    # versions = [1, 2, 3, 4, 5]

    # for with_hidden_var_bool in with_hidden_var_bools:
    #     for is_big_bool in is_big_bools:
    #         for max_parent_num in max_parent_nums:
    #             for version in versions:
    #                 kwargs = {
    #                     "with_hidden_var": with_hidden_var_bool,
    #                     "is_big": is_big_bool,
    #                     "max_parent_num": max_parent_num,
    #                     "version": version,
    #                 }
    #                 df = update_dataset_info("dataverse", kwargs, df)

    # versions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # names = [
    #     "Network2_amp",
    #     "Network3_amp",
    #     "Network4_amp",
    #     "Network5_amp",
    #     "Network5_cont",
    #     "Network5_cont_p3n7",
    #     "Network5_cont_p7n3",
    #     "Network6_amp",
    #     "Network6_cont",
    #     "Network7_amp",
    #     "Network7_cont",
    #     "Network8_amp_amp",
    #     "Network8_amp_cont",
    #     "Network8_cont_amp",
    #     "Network9_amp_amp",
    #     "Network9_amp_cont",
    #     "Network9_cont_amp",
    # ]
    # for name in names:
    #     for version in range(1, 61):
    #         if name == "Network6_amp" and version == 25:
    #             continue
    #         df = update_dataset_info(
    #             "simulated_feedback", {"name": name, "version": version}, df
    #         )

    # df = update_dataset_info("sachs", {}, df)

    # versions = [1, 2, 3, 4]
    # for version in versions:
    #     kwargs = {"version": version}
    #     df = update_dataset_info("dream4", kwargs, df)

    # sample_nums = [500, 1000, 5000]
    # versions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for sample_num in sample_nums:
    #     for version in versions:
    #         kwargs = {"sample_num": sample_num, "version": version}
    #         df = update_dataset_info("pigs", kwargs, df)

    # indexs = [1, 3, 5, 10]
    # sample_nums = [500, 1000, 5000]
    # versions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for index in indexs:
    #     for sample_num in sample_nums:
    #         for version in versions:
    #             kwargs = {"index": index, "sample_num": sample_num, "version": version}
    #             df = update_dataset_info("insurance", kwargs, df)

    df.to_csv(f"{dirname}/datasets_summary.csv", index=False)
