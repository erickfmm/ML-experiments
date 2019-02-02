import random
__all__ = ["random_assignment",]

#this code: import with_nolib.unsupervised.clustering.initial_assignments as ias

def random_assignment(X, num_clusters):
    assign = []
    for _ in range(len(X)):
        assign.append(random.randint(0, num_clusters-1))
    return assign

from utils.points_utils import distance


def assign_data_to_centroids(X, centroids, dist_fun=distance):
    assign = []
    for data in X:
        min_dist = float('+inf')
        assign_tmp = 0
        for icentroid in range(len(centroids)):
            dist_to_centroid = dist_fun(data, centroids[icentroid])
            if dist_to_centroid < min_dist:
                min_dist = dist_to_centroid
                assign_tmp = icentroid
        assign.append(assign_tmp)
    return assign

def furthest_mine(X, num_clusters):
    from statistics import stdev
    assign = []
    centroids = []
    distances = [[] for _ in range(num_clusters)]
    #choose randomly the first centroid
    centroids_indexes = []
    centroids_indexes.append(random.randint(0, len(X)-1 ))
    centroids.append(X[centroids_indexes[0]])
    distances[0] = [distance(centroids[0], el) for el in X]
    #for second new centroid
    #chooose the furthest point
    max_dist = 0.0
    i_maxdist = 0
    for idist in range(len(distances[0])):
        if distances[0][idist] > max_dist:
            max_dist = distances[0][idist]
            i_maxdist = idist
    centroids_indexes.append(i_maxdist)
    centroids.append(X[i_maxdist])
    distances[1] = [distance(centroids[1], el) for el in X]
    #for all the rest:
    for icentroid in range(2, num_clusters):
        min_dev = float('+inf')
        i_min_dev = 0
        for idist_data in range(len(distances[0])): #each data
            dists_ofdata = []
            for dist_centroid in distances:
                if len(dist_centroid) > 0 and idist_data not in centroids_indexes: # dist_centroid[idist_data] != 0:
                    dists_ofdata.append(dist_centroid[idist_data])
            if idist_data not in centroids_indexes and len(dists_ofdata) >=2 and stdev(dists_ofdata) <= min_dev:
                min_dev = stdev(dists_ofdata)
                i_min_dev = idist_data
        centroids_indexes.append(i_min_dev)
        centroids.append(X[i_min_dev])
        distances[icentroid] = [distance(centroids[icentroid], el) for el in X]
    return assign_data_to_centroids(X, centroids)