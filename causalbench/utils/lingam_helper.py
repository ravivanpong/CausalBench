"""_summary_
"""

import numpy as np


def from_weighted_to_ones_zeros(weighted):
    """_summary_
    current rule:
    convert everything other than 0 to 1

    Prerequisite:
    all nan in weighted are symmetric.
    It's not checked in this methid.

    Args:
        weighted (_type_): _description_
    """
    # weighted[np.isnan(weighted)] = 1  # convert all nan to 1
    return np.where(weighted == 0, 0, 1)
