from abc import ABC, abstractmethod

class BaseCastle(ABC):
    """Base class for all gCastle algorithms."""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def lib(self):
        pass
        
    @abstractmethod    
    def fit(self, data):
        # Subclasses should implement this method!
        pass