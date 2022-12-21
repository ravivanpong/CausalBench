"""
This file provides help functions for causal discovery.
"""


import logging
import numpy as np


def combine_two_lists(list_1, list_2):
    """
    Generate all possible combination of params from two lists
    Example:
    param_list_1 = ['a', 'b']
    param_list_2 = [True, False]
    combine_two_lists(param_list_1, param_list_2)
    The result will be [['a', True], ['a', False], ['b', True], ['b', False] ]
    """
    if list_1 == []:
        return list_2
    if list_2 == []:
        return list_1
    result_array = []
    for i in list_1:
        for j in list_2:
            if isinstance(i, list):
                temp = i.copy()
                temp.append(j)
                result_array.append(temp)
            else:
                result_array.append([i, j])
    return result_array


def combine_multiple_lists(lists):
    """Generate all possible combination of params from multiple lists."""
    if len(lists) < 2:
        return lists
    result_array = []
    for sublist in lists:
        result_array = combine_two_lists(result_array, sublist)
    return result_array


def load_data_from_cdt(dataset_name):
    """Load dataset from causal discovery toolbox

    Args:
        dataset_name (string): name of the dataset

    Returns:
         tuple: (pandas.DataFrame, pandas.DataFrame or networkx.DiGraph, numpy matrix)
    """
    from cdt.data import load_dataset
    from networkx import to_numpy_array

    data, true_graph = load_dataset(dataset_name)
    true_adj_matrix = np.asmatrix(to_numpy_array(true_graph))
    return data, true_graph, true_adj_matrix


def standardize_data(data):
    """Standardize dataset"""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(data)


def calc_varsortability(data, true_adj_matrix):
    """_summary_

    Args:
        data (_type_): _description_
        true_adj_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    from causalbench.metrics.varsortability import varsortability

    data_numpy = (
        data if isinstance(data, np.ndarray) else np.array(data)
    )  # varsortability accepts only np.array datatype

    try:
        varsort = varsortability(data_numpy, np.asarray(true_adj_matrix))
    except TypeError as err:
        logging.exception(err)
        return np.nan
    else:
        return varsort


def evaluate_cdt_metrics(estimated_adj_matrix, true_graph):
    """_summary_

    Args:
        estimated_adj_matrix (_type_): _description_
        true_graph (_type_): _description_

    Returns:
        _type_: _description_
    """

    shd = shd_cpdag = auc = curve = np.nan  ## Initialize metrics
    from cdt.metrics import SHD, SHD_CPDAG, precision_recall

    if estimated_adj_matrix.size != 0:
        try:
            shd = SHD(true_graph, estimated_adj_matrix, double_for_anticausal=True)
        except TypeError as err:
            logging.exception(err)

        try:
            shd_cpdag = SHD_CPDAG(true_graph, estimated_adj_matrix)
        except TypeError as err:
            logging.exception(err)

        try:
            auc, curve = precision_recall(true_graph, estimated_adj_matrix)
        except TypeError as err:
            logging.exception(err)

    return shd, shd_cpdag, auc, curve


def gen_output_file(path_result, file_name, dict_result):
    """_summary_

    Args:
        path_result (_type_): _description_
        file_name (_type_): _description_
        dict_result (_type_): _description_
    """
    import os.path
    from csv import DictWriter

    outfile = os.path.join(path_result, file_name)
    if os.path.exists(outfile):
        with open(outfile, "a", encoding="utf-8") as file:
            writer = DictWriter(file, dict_result.keys(), delimiter=";")
            writer.writerow(dict_result)
    else:
        with open(outfile, "w", encoding="utf-8") as file:
            writer = DictWriter(file, dict_result.keys(), delimiter=";")
            writer.writeheader()
            writer.writerow(dict_result)
