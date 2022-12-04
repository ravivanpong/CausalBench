
'''
Generate all possible combination of params from two lists
Example:
param_list_1 = ['a', 'b']
param_list_2 = [True, False]
combine_two_lists(param_list_1, param_list_2)
The result will be [['a', True], ['a', False], ['b', True], ['b', False] ]
'''
def combine_two_lists(list_1, list_2):
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

# generate all possible combination of params from multiple lists.
def combine_multiple_lists(lists):
    
    if len(lists) < 2:
        return lists
    result_array = []    
    for index in range(len(lists)):
        result_array = combine_two_lists(result_array, lists[index])
    return result_array  

def load_data_from_cdt(dataset_name):
    from cdt.data import load_dataset
    from networkx import to_numpy_matrix
    data, true_graph = load_dataset(dataset_name)
    true_adj_matrix = to_numpy_matrix(true_graph)
    return data, true_graph, true_adj_matrix

def standardize_data(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(data)

def calc_varsortability(data, true_adj_matrix):
    import numpy as np
    from metrics import varsortability
    try:
        if not isinstance(data, np.ndarray):  ## varsortability accepts only np.array datatype
            data_numpy = np.array(data)
        else:
            data_numpy = data
        varsort = varsortability(data_numpy, np.asarray(true_adj_matrix))
    except:
        varsort = np.nan
    return varsort

def evaluate_cdt_metrics(estimated_adj_matrix, true_graph):
    import numpy as np
    from cdt.metrics import SHD, SHD_CPDAG, precision_recall
    shd = shd_cpdag = auc = curve = np.nan  ## Initialize metrics

    if estimated_adj_matrix.size != 0:
        try:
            shd = SHD(true_graph, estimated_adj_matrix, double_for_anticausal=True)
        except Exception as e:
            print(e)

        try:
            shd_cpdag = SHD_CPDAG(true_graph, estimated_adj_matrix)
        except Exception as e:
            print(e)

        try:
            auc, curve = precision_recall(true_graph, estimated_adj_matrix)
        except Exception as e:
            print(e)

    return shd, shd_cpdag, auc, curve  

def gen_output_file(path_result, file_name, dict_result):
    import os.path
    from csv import DictWriter
    outfile = os.path.join(path_result, file_name)
    if os.path.exists(outfile):
        with open(outfile, "a", encoding="utf-8") as file:
            writer = DictWriter(file, dict_result.keys(), delimiter=';')
            writer.writerow(dict_result)
    else:
        with open(outfile, "w", encoding="utf-8") as file:
            writer = DictWriter(file, dict_result.keys(), delimiter=';')
            writer.writeheader()
            writer.writerow(dict_result) 
