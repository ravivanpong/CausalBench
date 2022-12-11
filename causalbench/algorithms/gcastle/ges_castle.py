"""_summary_
"""
import logging
import numpy as np
from castle.algorithms import GES
from causalbench.algorithms.base import Base


class GESCastle(GES, Base):
    """
    GESCastle uses GES from gCastle.
    It's subclass of GES and Base.
    """

    name = "GES"
    lib = "gCastle"

    def fit(self, data):

        try:
            self.learn(data)
        except TypeError as err:
            logging.exception(err)
            return np.array([])
        else:

            return self.causal_matrix
