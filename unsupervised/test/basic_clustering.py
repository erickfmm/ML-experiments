import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '../..')))
######################################################

import pprint
import numpy as np

import unsupervised.clustering.cluster_sklearn as clustering_sk

import unsupervised.clustering.utils.initial_assignments as ias
from unsupervised.clustering.kmeans import KMeans
from unsupervised.clustering.utils.monte_carlo import montecarlo_clustering
import unsupervised.clustering.utils.metrics as clustering_metrics

labels1 = [int(i/1000) for i in range(4000)]
np.random.seed(844)
clust1 = np.random.normal(5, 2, (1000,2))
clust2 = np.random.normal(15, 3, (1000,2))
clust3 = np.random.multivariate_normal([17,3], [[1,0],[0,1]], 1000)
clust4 = np.random.multivariate_normal([2,16], [[1,0],[0,1]], 1000)
simple_dataset = np.concatenate((clust1, clust2, clust3, clust4))




def evaluate_clustering(dataset1, n_clusters=4, original_labels = None, to_plot = False):
    all_assignments = {}
    if original_labels is not None:
        all_assignments["original"] = original_labels
    all_assignments["furthest"] = ias.furthest_mine(dataset1, n_clusters)
    all_assignments["random"] = ias.random_assignment(dataset1, n_clusters)

    km_model = KMeans(dataset1)
    km_model.initial_assignment = ias.random_assignment
    all_assignments["kmeans random"] = km_model.cluster(n_clusters, 300)

    km_model = KMeans(dataset1)
    km_model.initial_assignment = ias.furthest_mine
    all_assignments["kmeans furthest"] = km_model.cluster(n_clusters, 300)

    all_assignments["affinity"] = clustering_sk.affinity_propagation(dataset1, False)
    all_assignments["dbscan"] = clustering_sk.dbscan(dataset1)
    all_assignments["dirichlet"] = clustering_sk.dirichlet(dataset1, n_clusters)
    all_assignments["hierachical"] = clustering_sk.hierarchical(dataset1, n_clusters)
    all_assignments["gaussian mixture"] = clustering_sk.gaussian_mixture(dataset1, n_clusters)
    all_assignments["hierarchical connected"] = clustering_sk.hierarchical_connected(dataset1, n_clusters)
    all_assignments["kmeans"] = clustering_sk.kmeans(dataset1, n_clusters)
    all_assignments["kneighbours graph"] = clustering_sk.kneighbors_graph(dataset1, n_clusters)
    all_assignments["mean shift"] = clustering_sk.mean_shift(dataset1)
    
    pp = pprint.PrettyPrinter(indent=4)
    result_metrics = {}
    for assignment in all_assignments:
        result_metrics[assignment] = clustering_metrics.evaluate_all_metrics(dataset1, all_assignments[assignment])
    
    pp.pprint(result_metrics)
    if to_plot:
        #clustering_sk.plot_clusters(dataset1, original_labels)
        clustering_sk.plot_clusters(dataset1, all_assignments["furthest"])
        clustering_sk.plot_clusters(dataset1, all_assignments["random"])
    return result_metrics, all_assignments

evaluate_clustering(simple_dataset, 4, labels1, False)