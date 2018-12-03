from abc import ABCMeta, abstractmethod

class ILoadSupervised:
    __metaclass__ = ABCMeta
    TYPE = "supervised"

    @classmethod
    def version(self): return "1.0"
    
    @abstractmethod
    def get_default(self): raise NotImplementedError

    @abstractmethod
    def get_splited(self): raise NotImplementedError

    @abstractmethod
    def get_all(self): raise NotImplementedError