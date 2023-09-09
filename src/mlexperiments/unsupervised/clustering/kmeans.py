import random
import math
from mlexperiments.unsupervised.clustering.ICluster import ICluster
from mlexperiments.unsupervised.clustering.utils.initial_assignments import random_assignment

__all__ = ["KMeans"]


class KMeans(ICluster):
    def __init__(self, data, initial_assignment_function=random_assignment):
        super().__init__(data)
        self.X = data
        self.assign = []
        self.centroids = []
        self.distances = []
        self.initial_assignment = initial_assignment_function
        self.iterations = 0
        # for i in range(len(data)):
        #    self.assign.append(-1)

    def create_centroids(self, num_clusters):
        for i_centroid in range(num_clusters):
            self.centroids.append([])
            for i_data in range(len(self.X[0])):
                self.centroids[i_centroid].append(0)

    def compute_centroids(self):
        num_data_centroids = []
        for i_centroid in range(len(self.centroids)):
            num_data_centroids.append(0)
            # restart centroids
            for i_data in range(len(self.centroids[i_centroid])):
                self.centroids[i_centroid][i_data] = 0
        for i in range(len(self.X)):
            num_data_centroids[self.assign[i]] += 1
            for i_data in range(len(self.X[i])):
                self.centroids[self.assign[i]][i_data] += self.X[i][i_data]
        # now make average
        for i_centroid in range(len(self.centroids)):
            # restart centroids
            for i_data in range(len(self.centroids[i_centroid])):
                if self.centroids[i_centroid][i_data] != 0 and num_data_centroids[i_centroid] != 0:
                    self.centroids[i_centroid][i_data] = self.centroids[i_centroid][i_data] \
                    / float(num_data_centroids[i_centroid])

    def reassign(self):
        for i in range(len(self.X)):
            min_distance = float('+inf')
            min_centroid = -1
            for i_centroid in range(len(self.centroids)):
                distance = 0
                for i_data in range(len(self.X[i])):
                    distance += (self.X[i][i_data] - self.centroids[i_centroid][i_data])**2
                distance = math.sqrt(distance)
                self.distances[i][i_centroid] = distance
                if distance < min_distance:
                    min_distance = distance
                    min_centroid = i_centroid
            self.assign[i] = min_centroid

    def cluster(self, num_clusters, max_iterations=100):
        self.create_centroids(num_clusters)
        self.assign = self.initial_assignment(self.X, num_clusters)
        for i in range(len(self.X)):
            self.distances.append([])
            for i_centroid in range(num_clusters):
                self.distances[i].append(0)
        self.iterations = 0
        while True:
            # clone list
            last_assignment = list(self.assign)
            self.compute_centroids()
            self.reassign()
            if last_assignment == self.assign:
                break
            if 0 < max_iterations <= self.iterations:
                break
            self.iterations += 1
        return self.assign


# randomly assign all data items to a cluster
# loop until no change in cluster assignments
#    compute centroids for each cluster
#    reassign each data item to cluster of closest centroid
# end