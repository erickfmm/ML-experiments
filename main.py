from tests import perceptron
from load_data.data_inside.shared import *
from with_nolib.unsupervised.clustering.kmeans import KMeans
from with_nolib.unsupervised.clustering.monte_carlo import montecarlo_clustering
import numpy as np

#data = LoadAndTable()
#data = LoadXorTable()
data = LoadTitanic()
X, Y = data.get_default()
#perceptron.train(X, Y)

#X = []
#for i in range(200):
    #X.append(list(np.random.normal(0, 10, 36)))
    #X.append(list(np.random.random(5)))
print("len: ", len(X))
#print("Y: ", np.sum(Y))
#print("not Y: ", len(Y)-np.sum(Y))
f, a, ms = montecarlo_clustering(KMeans, X, 10, 3, 30)
#km = KMeans(X)
#km.cluster(2, 0)
#print("centroids: ", km.centroids)
print("assign 0: ", np.sum(np.isin(f,0)))
print("assign 1: ", np.sum(np.isin(f,1)))
print("assign 2: ", np.sum(np.isin(f,2)))
#print("assign 3: ", np.sum(np.isin(km.assign,3)))
for m in ms:
    print("its: ", m.iterations)
#print("Xs: ", km.X)
#print("distances: ", km.distances)