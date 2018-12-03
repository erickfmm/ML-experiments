from tests import perceptron
from load_data.data_inside.shared import *
from with_nolib.unsupervised.kmeans import KMeans
import numpy as np

#data = LoadAndTable()
#data = LoadXorTable()
data = LoadTitanic()
X, Y = data.get_default()
#perceptron.train(X, Y)

print("len: ", len(X))
print("Y: ", np.sum(Y))
print("not Y: ", len(Y)-np.sum(Y))
km = KMeans(X)
km.cluster(2)
#print("centroids: ", km.centroids)
print("assign 0: ", np.sum(np.isin(km.assign,0)))
print("assign 1: ", np.sum(np.isin(km.assign,1)))
print("assign 2: ", np.sum(np.isin(km.assign,2)))
print("assign 3: ", np.sum(np.isin(km.assign,3)))
print("its: ", km.iterations)
#print("Xs: ", km.X)
#print("distances: ", km.distances)