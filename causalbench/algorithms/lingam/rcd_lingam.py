"""_summary_

Returns:
    _type_: _description_
"""
import logging
import numpy as np
from lingam import RCD
from causalbench.algorithms.base import Base

import causalbench.utils.lingam_helper as helper


class RCDLingam(RCD, Base):
    """_summary_

    Args:
        RCD (_type_): _description_
        Base (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        max_explanatory_num=2,
        cor_alpha=0.01,
        ind_alpha=0.01,
        shapiro_alpha=0.01,
        MLHSICR=False,
        bw_method="mdbs",
    ):
        super().__init__(
            max_explanatory_num, cor_alpha, ind_alpha, shapiro_alpha, MLHSICR, bw_method
        )
        self.ones_zeros_matrix = None

    name = "RCD"
    lib = "Lingam"

    def fit(self, data):

        try:
            super().fit(data)
        except TypeError as err:
            logging.exception(err)
            return np.array([])
        else:
            self.ones_zeros_matrix = helper.from_weighted_to_ones_zeros(
                self.adjacency_matrix_
            )
            return self.ones_zeros_matrix
