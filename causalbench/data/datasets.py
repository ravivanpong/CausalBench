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

import pandas as pd
from .metrics.varsortability import varsortability



def list_datasets(self):
    """
    List all the datasets avaialble in the packages
    """

    
def summarize(dataset='all', file=None, update=None):
    """
    Give an overview summary of a given or all the available datasets

    Returns
    --------
    data_summary : pd.DataFrame : contains 
        [dataset name (variant name), number of variables, number of samples, varsortability]
    """

    ## If a output file path is given, then use it

    

