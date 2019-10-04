from abc import ABCMeta, abstractmethod, ABC

from enum import Enum
class SupervisedType(Enum):
    Unknown = 0
    Classification = 1
    Regression = 2
    Both = 3

class ILoadSupervised(ABC):
    __metaclass__ = ABCMeta
    TYPE: SupervisedType = SupervisedType.Unknown
    
    #@classmethod
    #def version(self): return "1.0"
    
    @abstractmethod
    def get_all(self): raise NotImplementedError
    
    #@abstractmethod
    def get_all_yielded(self): raise NotImplementedError
    
    @abstractmethod
    def get_classes(self): raise NotImplementedError
    
    @abstractmethod
    def get_headers(self): raise NotImplementedError


class ISplitted(ABC):
    @abstractmethod
    def get_splited(self): raise NotImplementedError

    #@abstractmethod
    def get_train_yielded(self): raise NotImplementedError

    #@abstractmethod
    def get_test_yielded(self): raise NotImplementedError

    #@abstractmethod
    def get_train(self): raise NotImplementedError

    #@abstractmethod
    def get_test(self): raise NotImplementedError