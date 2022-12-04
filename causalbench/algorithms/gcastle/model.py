from abc import ABC, abstractmethod

class Model(ABC):
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
    def fit(self, data, parameters):
        pass