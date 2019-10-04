from abc import ABCMeta, abstractmethod, ABC

class IMetadata(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_metadata(self): raise NotImplementedError

    @abstractmethod
    def get_metadata_headers(self): raise NotImplementedError