"""_summary_

Returns:
    _type_: _description_
"""
import logging
import numpy as np
from castle.algorithms import ANMNonlinear
from causalbench.algorithms.base import Base


class ANMCastle(ANMNonlinear, Base):
    """
    ANMCastle uses ANMNonliner from gCastle.
    It's subclass of ANMNonliner and Base.
    """

    name = "ANM"
    lib = "gCastle"

    def fit(self, data):

        try:
            self.learn(data)
        except TypeError as err:
            logging.exception(err)
            return np.array([])
        else:
            return self.causal_matrix
