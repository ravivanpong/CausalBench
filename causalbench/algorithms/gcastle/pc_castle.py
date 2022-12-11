"""_summary_
"""
import logging
import numpy as np
from castle.algorithms import PC
from causalbench.algorithms.base import Base


class PCCastle(PC, Base):
    """
    PCCastle uses PC from gCastle.
    It's subclass of PC and Base.
    """

    name = "PC"
    lib = "gCastle"

    def fit(self, data):

        try:
            self.learn(data)
        except TypeError as err:
            logging.exception(err)
            return np.array([])
        else:

            return self.causal_matrix
