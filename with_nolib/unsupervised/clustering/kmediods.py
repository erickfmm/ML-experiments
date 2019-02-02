"""
The most common realisation of k-medoid clustering
is the Partitioning Around Medoids (PAM) algorithm.
PAM uses a greedy search which may not find the optimum solution,
 but it is faster than exhaustive search. It works as follows:

Initialize: select k of the n data points as the medoids
Associate each data point to the closest medoid.
While the cost of the configuration decreases:
For each medoid m, for each non-medoid data point o:
Swap m and o, associate each data point to the closest medoid,
    recompute the cost (sum of distances of points to their medoid)
If the total cost of the configuration increased in the previous step, undo the swap

The runtime complexity of the original PAM algorithm per iteration of (3)
 is O(k(n-k)^{2}), but this can be reduced to O(n^{2}).[2]

Algorithms other than PAM have also been suggested in the literature,
including the following Voronoi iteration method:[3][4]

Select initial medoids
Iterate while the cost decreases:
In each cluster, make the point that minimizes the sum of distances within the cluster the medoid
Reassign each point to the cluster defined by the closest medoid determined in the previous step.

However, Voronoi iteration finds much worse results,
as it does not allow reassigning points to other clusters while changing means,
and thus only explores a much smaller search space.[2]"""