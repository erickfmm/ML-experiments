from abc import ABCMeta, abstractmethod
from with_nolib.unsupervised.clustering.initial_assignments import *

class ICluster:
    __metaclass__ = ABCMeta
    def __init__(self, data):
        self.X = data
        self.assign = []
        self.centroids = []
        self.initial_assignment = random_assignment

    @abstractmethod
    def cluster(self, num_clusters, max_iterations): raise NotImplementedError