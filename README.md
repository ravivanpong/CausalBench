# CausalBench
Causal discovery is a subfield of causal inference that aims to retrieve a causal graph from quantitative data. As in machine learning, there is no single algorithm that performs well in all scenarios. However, given the large number of available algorithms and ever increasing new ones being published quarterly, each claiming to be better than the state-of-the-art based on selected metrics, practitioners find the choices overwhelming. Since there is no ground truth for model selection and hyperparameter optimization, causal discovery algorithm selection is based on data assumptions, its sensitivity, if these assumptions are violated, and experience. The insight to the later two can be achieved through comparison and benchmark studies. 

## Goals
* Provide an overview of existing causal discovery packages, their differences, and their development status
* Provide an overview of algorithm performance under different scenarios
* Produce a data-driven guideline for causal discovery algorithm selection

## Preliminary results
* See (/causalbench/data)[/causalbench/data] for list of included data
* See Bachelor thesis (Evaluation and Comparison of Causal Discovery Algorithms)[documentation\BachelorThesis_Evaluation-and -Comparison-of-Causal-Discovery-Algorithms.pdf] 
* Analysis of surveyed packages (coming soon)
* Overview of algorithms and their implementations (coming soon)

## Method
* Survey existing causal discovery packages.
* Collect datasets (real world, semi-synthetic, synthetic) with "ground truth" and create a unified data format for experiment.  
* Collect single, original implementations of published algorithms that are not yet included in known packages.  
* Evaluate large number of existing causal discovery algorithms for cross-sectional data (i.e. not time series) on various scenarios.


## Authors
* Ployplearn Ravivanpong (ravivanpong@teco.edu, p.vohr.ravivanpong@mail.de)
* Hui Gong

## Acknowlegement
Part of this research is funded by SDSC-BW program.

Many thanks for the Daniel Thaden for helping in the search for the datasets and repositories of single implementations. We fully appriciate Maximilian Winkler, who found and tested additional algorithms.

## References
TBD

## Disclaimer
We tried to include as many algorithms as possible that are published up to the end of 2022. Nevertheless, due to time and resources, we are not able to find each and every algorithm ever published and / or implemented nor are we able to include every implementation in the experiment due to bugs and compatibility issues. We also try to keep up with data description and documentation as much as possible. If we use your data, but the source is not yet given, please remind us about it.
