"""_summary_

Returns:
    _type_: _description_
"""
import logging
import numpy as np
from lingam import DirectLiNGAM
from causalbench.algorithms.base import Base

import causalbench.utils.lingam_helper as helper


class DirectLiNGAMLingam(DirectLiNGAM, Base):
    """_summary_

    Args:
        DirectLiNGAM (_type_): _description_
        Base (_type_): _description_
    """

    def __init__(
        self,
        random_state=None,
        prior_knowledge=None,
        apply_prior_knowledge_softly=False,
        measure="pwling",
    ):
        super().__init__(
            random_state, prior_knowledge, apply_prior_knowledge_softly, measure
        )
        self.causal_matrix = None

    name = "DirectLiNGAM"
    lib = "Lingam"

    def fit(self, data):

        try:
            super().fit(data)
        except TypeError as err:
            logging.exception(err)
            return np.array([])
        except ValueError as verr:
            logging.exception(verr)
            return np.array([])
        else:
            self.causal_matrix = helper.from_weighted_to_ones_zeros(
                self.adjacency_matrix_
            )
            return self.causal_matrix
