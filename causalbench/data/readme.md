# Causal Discovery Datasets
The data sets collected here are real world, semi-synthetic, and synthetic datasets for benchmarking causal discovery methods from different sources. The repository contains data sets and data loaders for each data set. There may be different versions of the datasets with the same name published or used elsewhere. If you are aware or have them, feel free to contact us or share them with us.

**NOTE : A data set is added only if it contains a ground-truth graph.**

## Folder structure
Each group of data set has its own folder. Each folder is named according to the name the dataset is typically known or called by its providers. 
```
Data set name
  |   short-dataset-name_version-number_data.csv
  |   short-dataset-name_version-number_target.csv
  |   short-dataset-name_graph.csv
  |   short-dataset-name_info.txt
  |   load.py
```
* `*_data.csv` contains the data used for causal discovery
* `*_target.csv` contains (directed) edge list of the true graph. The entry of each row can be read as `[source target (weight, if available)]`
* `*_graph.csv` alternative to `*_target.csv`, contains (weighted) adjacency matrix of the true graph
* `*_info.txt` contains data description. We do not create these descriptions by ourselves for every data set, but copy the original description and adapt them to our format for ease of reading. References to the original data source and descriptions are given.
* `load.py` a data loader script specific for the dataset(s) in the folder. The data loader returns a dataset or a sub-dataset and their corresponding ground truth graph. By default, the ground truth graph is not weighted. Using the option `weighted = True` will return the weighted ground truth graph if available. Otherwise, a warning will be given and the unweighted ground truth graph is returned.

## Data description
A data description file is provided in each data folder.  

## Related works
The template for data cards and data loaders are inspired from 
* causal-discovery-toolbox data loader (https://github.com/FenTechSolutions/CausalDiscoveryToolbox)


## Currently included datasets
### Real world
* JDK
* Networking
* PostgresSQL

### Semi-Synthetic
* ALARM
* DREAM4

### Synthetic
* 

