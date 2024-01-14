import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

from mlexperiments.load_data.loader.basic.random_of_function import LoadRandom

import mlexperiments.unsupervised.distance as distance
import mlexperiments.unsupervised.correlation_matrix as correlation_matrix
import mlexperiments.supervised.utils.confusion_matrix as confusion_matrix

from mlexperiments.supervised.perceptron import Perceptron
import mlexperiments.supervised.learning_models_sklearn as learning_model

import random
import math

def print2d(data):
    for row in data:
        for n in row:
            print(round(n, 3), end="\t")
        print()
    print()

class RandomClass:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
    def __call__(self, x):
        return random.randint(0,self.n_classes-1)

class TanhClass:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
    def __call__(self, x):
        tag = math.floor(math.tanh((math.fsum(x)/len(x)) )* (self.n_classes))
        return tag if tag >= 0 and tag < self.n_classes else self.n_classes-1


random3 = RandomClass(3)
tanh3 = TanhClass(3)
print("to load...")
lrand = LoadRandom(num_instances=1000, num_dimensions=5, min_value=0, max_value=1, fn=tanh3)
x, y = lrand.get_X_Y()

print("some calculations")
rs = correlation_matrix.correlation_matrix(x)
print("correlations")
print2d(rs)
dists = distance.distance_matrix(x)
print("distances")
print2d(dists)
ms, bs = correlation_matrix.coefficients_matrix(x)
print("ms")
print2d(ms)
print("bs")
print2d(bs)

print()
print("to learn....")
print()
model, clf = learning_model.train_mlp(x, y)
#pr = Perceptron()
#pr.train(x, y)
scores = learning_model.get_k_scores(clf, x, y)
print("scores: ", scores)
print("mean scores: ", sum(scores)/10.0)

#ypreds = pr.classify(x)
ypreds = model.predict(x)
conf = confusion_matrix.confusion_matrix(y, ypreds)
print("confusion matrix")
print2d(conf)

conf_norm = confusion_matrix.normalize(conf)
print("normalized")
print2d(conf_norm)