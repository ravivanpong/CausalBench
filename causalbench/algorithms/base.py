"""
This Module defines base modle for all algorithms in this framework.
All algorithm should have attribute 'lib' indicating which libriry it's imported from.
All algorithm should have attribute 'name' indicating its common name.
All algorithm should have method 'fit()' to estimate causal matrix.
"""

from abc import ABC, abstractmethod


class Base(ABC):
    """Base class for all algorithms."""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        """Common name of this algorithm"""

    @property
    @abstractmethod
    def lib(self):
        """lib indicates which library this algorithm comes from"""

    @abstractmethod
    def fit(self, data):
        """
        Subclasses should implement this method.
        This method takes in data and returns estimated causal matrix.
        """
