from abc import ABCMeta, abstractmethod

from enum import Enum
class SupervisedType(Enum):
    Unknown = 0
    Classification = 1
    Regression = 2
    Both = 3

class ILoadSupervised:
    __metaclass__ = ABCMeta
    TYPE: SupervisedType = SupervisedType.Unknown
    #headers = []
    #classes = []

    @classmethod
    def version(self): return "1.0"
    
    @abstractmethod
    def get_default(self): raise NotImplementedError

    @abstractmethod
    def get_splited(self): raise NotImplementedError

    @abstractmethod
    def get_all(self): raise NotImplementedError
    
    @abstractmethod
    def get_classes(self): raise NotImplementedError
    
    @abstractmethod
    def get_headers(self): raise NotImplementedError