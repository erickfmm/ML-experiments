from abc import ABCMeta, abstractmethod
import unsupervised.clustering.utils.initial_assignments as init_assign


class ICluster:
    __metaclass__ = ABCMeta

    def __init__(self, data):
        self.X = data
        self.assign = []
        self.centroids = []
        self.initial_assignment = init_assign.random_assignment

    @abstractmethod
    def cluster(self, num_clusters, max_iterations): raise NotImplementedError
