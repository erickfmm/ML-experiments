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

    @abstractmethod
    def get_X_Y(self): raise NotImplementedError

    @abstractmethod
    def get_classes(self): raise NotImplementedError
    
    @abstractmethod
    def get_headers(self): raise NotImplementedError
