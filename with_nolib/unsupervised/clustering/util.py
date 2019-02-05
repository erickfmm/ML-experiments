import utils.points_utils as putils

def generate_mean_centroids(X, assignments):
    n_centroids = len(set(assignments))
    centroids = [ [] for _ in range(n_centroids) ]
    n_data_centroid = [0 for _ in range(n_centroids)]
    for icentroid in range(len(centroids)):
        for _ in range(len(X[0])): #ndim
            centroids[icentroid].append(0)
    for iX in range(len(X)):
        cluster_assigned = assignments[iX]
        n_data_centroid[cluster_assigned] += 1
        for ifeat in range(len(X[iX])):
            centroids[cluster_assigned][ifeat] += X[iX][ifeat]
    for icentroid in range(len(centroids)):
        for idim in range(len(centroids[icentroid])): #dims
            centroids[icentroid][idim] /= float(n_data_centroid[icentroid])
    return centroids