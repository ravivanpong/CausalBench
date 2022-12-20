"""_summary_

Returns:
    _type_: _description_
"""
import logging
import numpy as np
from castle.algorithms.gradient.pnl.torch import PNL
from causalbench.algorithms.base import Base


class PNLCastle(PNL, Base):
    """_summary_

    Args:
        PNL (_type_): _description_
        Base (_type_): _description_

    Returns:
        _type_: _description_
    """

    name = "PNL"
    lib = "gCastle"

    def fit(self, data):

        try:
            self.learn(data)
        except TypeError as terr:
            logging.exception(terr)
            return np.array([])
        except ValueError as verr:
            logging.exception(verr)
            return np.array([])
        else:
            return self.causal_matrix
