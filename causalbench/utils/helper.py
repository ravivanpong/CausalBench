"""
This file provides help functions for conducting experiments.
"""


import logging
import numpy as np
import os.path
from csv import DictWriter
from causalbench.metrics.varsortability import varsortability


def init_func_with_param(func, kwargs: dict):
    """_summary_

    Args:
        func (_type_): _description_
        kwargs (dict): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if bool(kwargs):
        return func(**{k: v for k, v in kwargs.items() if v is not None})
    else:
        raise ValueError("kwargs can not be empty.")


def load_datasest(name: str, kwargs: dict):
    """_summary_

    Args:
        name (str): _description_
        kwargs (dict, optional): _description_. Defaults to {}.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if name.lower() == "alarm":
        from causalbench.data.alarm.alarm_loader import load_alarm

        return init_func_with_param(load_alarm, kwargs)
    elif name.lower() == "dream4":
        from causalbench.data.dream4.dream4_loader import load_dream4

        return init_func_with_param(load_dream4, kwargs)
    elif name.lower() == "jdk":
        from causalbench.data.jdk.jdk_loader import load_jdk

        return load_jdk()
    elif name.lower() == "postgres":
        from causalbench.data.postgres.postgres_loader import load_postgres

        return load_postgres()
    elif name.lower() == "sachs":
        from causalbench.data.sachs.sachs_loader import load_sachs

        return load_sachs()
    elif name.lower() == "networking":
        from causalbench.data.networking.networking_loader import load_networking

        return load_networking()
    elif name.lower() == "real_yacht":
        from causalbench.data.real_yacht.real_yacht_loader import load_real_yacht

        return load_real_yacht()
    elif name.lower() == "real_cites":
        from causalbench.data.real_cites.real_cites_loader import load_real_cites

        return load_real_cites()
    elif name.lower() == "real_auto_mpg":
        from causalbench.data.real_auto_mpg.real_auto_mpg_loader import (
            load_real_auto_mpg,
        )

        return load_real_auto_mpg()
    elif name.lower() == "simulated_feedback":
        from causalbench.data.simulated_feedback.simulated_feedback_loader import (
            load_feedback,
        )

        return init_func_with_param(load_feedback, kwargs)
    elif name.lower() == "child":
        from causalbench.data.child.child_loader import load_child

        return init_func_with_param(load_child, kwargs)
    elif name.lower() == "insurance":
        from causalbench.data.insurance.insurance_loader import load_ins

        return init_func_with_param(load_ins, kwargs)
    elif name.lower() == "hailfinder":
        from causalbench.data.hailfinder.hailfinder_loader import load_hailf

        return init_func_with_param(load_hailf, kwargs)
    elif name.lower() == "barley":
        from causalbench.data.barley.barley_loader import load_barley

        return init_func_with_param(load_barley, kwargs)
    elif name.lower() == "mildew":
        from causalbench.data.mildew.mildew_loader import load_mildew

        return init_func_with_param(load_mildew, kwargs)
    elif name.lower() == "munin1":
        from causalbench.data.munin1.munin1_loader import load_munin1

        return init_func_with_param(load_munin1, kwargs)
    elif name.lower() == "pigs":
        from causalbench.data.pigs.pigs_loader import load_pigs

        return init_func_with_param(load_pigs, kwargs)
    elif name.lower() == "link":
        from causalbench.data.link.link_loader import load_link

        return init_func_with_param(load_link, kwargs)
    elif name.lower() == "gene":
        from causalbench.data.gene.gene_loader import load_gene

        return init_func_with_param(load_gene, kwargs)
    elif name.lower() == "dataverse":
        from causalbench.data.dataverse.dataverse_loader import load_dataverse

        return init_func_with_param(load_dataverse, kwargs)
    else:
        raise ValueError(
            f"Data set: {name} with {kwargs} not found. Please check info.txt for supported datasets."
        )


def _combine_two_lists(list_1, list_2):
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
        result_array = _combine_two_lists(result_array, sublist)
    return result_array


def standardize_data(data):
    """Standardize dataset"""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(data)


def calc_varsortability(data, true_adj_matrix):
    """_summary_

    Args:
        data (_type_): _description_s
        true_adj_matrix (_type_): _description_

    Returns:
        _type_: _description_
    """

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


def dataframe_to_edges(df, source_name, target_name) -> list:
    edges = []
    for _, row in df.iterrows():
        edges.append((row[source_name], row[target_name]))
    return edges


def edges_to_matrix(edges: list, nodes: list):
    node_index_map = {}
    i = 0
    for node in nodes:
        node_index_map[node] = i
        i += 1

    n_nodes = len(nodes)

    matrix = np.zeros((n_nodes, n_nodes))
    for edge in edges:
        matrix[node_index_map[edge[0]]][node_index_map[edge[1]]] = 1
    return matrix
