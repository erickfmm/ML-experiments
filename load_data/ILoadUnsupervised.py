from abc import ABCMeta, abstractmethod, ABC


class ILoadUnsupervised(ABC):
    __metaclass__ = ABCMeta
    TYPE = "unsupervised"
    
    @abstractmethod
    def get_headers(self): raise NotImplementedError

    @abstractmethod
    def get_all(self): raise NotImplementedError
