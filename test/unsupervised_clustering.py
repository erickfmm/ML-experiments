import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))

from supervised.perceptron import Perceptron

# Load data
# data shared
from load_data.loader.basic.andtable import LoadAndTable
from load_data.loader.basic.xortable import LoadXorTable
from load_data.loader.basic.titanic import LoadTitanic

from load_data.loader.basic.mnist_file import LoadMnist
from load_data.loader.recognition_human_actions_video import LoadRecognitionHumanActions


from load_data.loader.downloadable.mnist_keras import LoadMnist
from load_data.loader.downloadable.iris_sklearn import LoadIris
from load_data.loader.downloadable.cifar10_keras import LoadCifar10

from unsupervised.clustering.kmeans import KMeans
from unsupervised.clustering.utils.monte_carlo import montecarlo_clustering
import numpy as np

#data = LoadAndTable()
data = LoadXorTable()
#data = LoadTitanic()
X, Y = data.get_all()

#pr = Perceptron()
#performance = pr.train(X, Y)
#print(performance)

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
print("assign 0: ", np.sum(np.isin(f, 0)))
print("assign 1: ", np.sum(np.isin(f, 1)))
print("assign 2: ", np.sum(np.isin(f, 2)))
#print("assign 3: ", np.sum(np.isin(km.assign,3)))
for m in ms:
    print("its: ", m.iterations)
#print("Xs: ", km.X)
#print("distances: ", km.distances)