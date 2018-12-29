from abc import ABCMeta, abstractmethod
from with_nolib.unsupervised.clustering.initial_assignments import *

class ICluster:
    __metaclass__ = ABCMeta
    X = []
    assign = []
    centroids = []
    initial_assignment = random_assignment

    @abstractmethod
    def cluster(self, num_clusters, max_iterations): raise NotImplementedError