"""_summary_

Returns:
    _type_: _description_
"""
import logging
import numpy as np
import pandas as pd
from castle.algorithms import DirectLiNGAM
from causalbench.algorithms.base import Base


class DirectLiNGAMCastle(Base):
    """
    Summary
    """

    def __init__(self, prior_knowledge=None, measure="pwling", thresh=0.3):
        self.prior_knowledge = prior_knowledge
        self.measure = measure
        self.thresh = thresh
        self.causal_matrix = None
        self.weight_causal_matrix = None

    name = "DirectLiNGAM"
    lib = "gCastle"

    def fit(self, data):
        logging.info("calling fit from direct lingam castle")
        prior_knowledge = self.prior_knowledge
        measure = self.measure
        thresh = self.thresh

        lingam_instance = DirectLiNGAM(
            prior_knowledge=prior_knowledge, measure=measure, thresh=thresh
        )
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        try:
            lingam_instance.learn(data)

        except TypeError as err:
            logging.exception(err)
            return np.array([])
        except ValueError as verr:
            logging.exception(verr)
            return np.array([])
        else:
            self.causal_matrix = lingam_instance.causal_matrix
            self.weight_causal_matrix = lingam_instance.adjacency_matrix_.T

            return self.causal_matrix
