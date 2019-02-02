from abc import ABCMeta, abstractmethod, ABC

class ILoadUnsupervised(ABC):
    __metaclass__ = ABCMeta
    TYPE = "unsupervised"

    @classmethod
    def version(self): return "1.0"
    
    @abstractmethod
    def get_headers(self): raise NotImplementedError

    @abstractmethod
    def get_default(self): raise NotImplementedError

    @abstractmethod
    def get_all(self): raise NotImplementedError