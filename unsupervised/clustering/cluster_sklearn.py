from enum import Enum
from sklearn import cluster, mixture #, datasets
from sklearn.neighbors import kneighbors_graph
# import numpy as np
# from sklearn.metrics import silhouette_score

# this code: import with_lib.unsupervised.cluster_sklearn as csk


class ClusterAlgorithm(Enum):
    KMeans = 1
    AgglomerativeClustering = 2  # hierarchical
    GaussianMixture = 3
    MeanShift = 4
    AffinityPropagation = 5
    DBSCAN = 6


def kmeans(dataset, n_clusters, max_iter=300, init='k-means++', n_init=10):
    return cluster.KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init, n_init=n_init).fit_predict(dataset)


def hierarchical(dataset, n_clusters=4, affinity='euclidean', linkage='ward'):
    return cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity,
                                           linkage=linkage).fit_predict(dataset)


def hierarchical_connected(dataset, n_clusters, n_neighbors=5, include_self=False,
                           affinity='euclidean', linkage='complete'):
    connect = kneighbors_graph(dataset, n_neighbors=n_neighbors, include_self=include_self)
    return cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity,
                                           linkage=linkage,connectivity=connect).fit_predict(dataset)


def gaussian_mixture(dataset, n_clusters, covariance_type='full'):
    return mixture.GaussianMixture(n_components=n_clusters, covariance_type=covariance_type).fit(dataset)


def dirichlet(dataset, n_clusters, max_iter=100, covariance_type='full',
              weight_concentration_prior_type='dirichlet_process'):
    # covariance_type : {‘full’, ‘tied’, ‘diag’, ‘spherical’}, defaults to 'full'
    # init_params : {‘kmeans’, ‘random’}, defaults to ‘kmeans’.
    # weight_concentration_prior_type:
    #   #'dirichlet_process' (using the Stick-breaking representation),
    #   #'dirichlet_distribution' (can favor more uniform weights).
    return mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type=covariance_type,
                                           weight_concentration_prior_type=weight_concentration_prior_type,
                                           max_iter=max_iter).fit_predict(dataset)


def mean_shift(dataset, quantile=0.1, bin_seeding=True):
    band = cluster.estimate_bandwidth(dataset, quantile=quantile)
    meanshift = cluster.MeanShift(bandwidth=band, bin_seeding=bin_seeding).fit(dataset)
    return meanshift.predict(dataset)


def affinity_propagation(dataset, verbose=True):
    return cluster.AffinityPropagation(verbose=verbose).fit_predict(dataset)


def dbscan(dataset, eps=1, min_samples=5, metric='euclidean'):
    return cluster.DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(dataset)


##
def plot_clusters(set1, colours1='gray', title1='Dataset 1'):
    import matplotlib.pyplot as plt
    fig,(ax1) = plt.subplots(1, 1)
    fig.set_size_inches(6, 3)
    ax1.set_title(title1, fontsize=14)
    ax1.set_xlim(min(set1[:, 0]), max(set1[:, 0]))
    ax1.set_ylim(min(set1[:, 1]), max(set1[:, 1]))
    ax1.scatter(set1[:, 0], set1[:, 1], s=8, lw=0, c=colours1)
    fig.tight_layout()
    plt.show()


def plots2_clusters(set1, set2, colours1='gray', colours2='gray',
                    title1='Dataset 1',  title2='Dataset 2'):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    ax1.set_title(title1, fontsize=14)
    ax1.set_xlim(min(set1[:, 0]), max(set1[:, 0]))
    ax1.set_ylim(min(set1[:, 1]), max(set1[:, 1]))
    ax1.scatter(set1[:, 0], set1[:, 1], s=8, lw=0, c=colours1)
    ax2.set_title(title2, fontsize=14)
    ax2.set_xlim(min(set2[:, 0]), max(set2[:, 0]))
    ax2.set_ylim(min(set2[:, 1]), max(set2[:, 1]))
    ax2.scatter(set2[:, 0], set2[:, 1], s=8, lw=0, c=colours2)
    fig.tight_layout()
    plt.show()
