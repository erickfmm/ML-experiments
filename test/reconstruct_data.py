import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))
######################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

from unsupervised.autoencoder_tf import reconstruct_data

#from load_data.loader.basic.iris import LoadIris
from load_data.loader.downloadable.iris_sklearn import LoadIris
from load_data.loader.downloadable.mnist_keras import LoadMnist
from load_data.loader.basic.glass import LoadGlass
from load_data.loader.basic.segment import LoadSegment
from load_data.loader.basic.titanic import LoadTitanic
from load_data.loader.basic.vehicle import LoadVehicle

print("all imported")


def calc_differences(real_data, reconstructed_data):
    difs = np.abs(real_data - reconstructed_data)
    difs_percentage = difs / real_data
    mean_squared = 0.0
    mean_difs = 0.0
    mean_difs_percentage = 0.0
    for idata in range(len(real_data)):
        for idim in range(len(real_data[0])):
            mean_squared += (real_data[idata][idim] - reconstructed_data[idata][idim])**2
            mean_difs += difs[idata][idim]
            mean_difs_percentage += difs_percentage[idata][idim]
    mean_squared /= (len(real_data) * len(real_data[0]))
    mean_difs /= (len(real_data) * len(real_data[0]))
    mean_difs_percentage /= (len(real_data) * len(real_data[0]))
    rmse = np.sqrt(mean_squared)
    print("rmse: ", rmse)
    print("mean difs: ", mean_difs)
    print("mean difs percentage: ", mean_difs_percentage)
    print("mean squared: ", mean_squared)


l_iris = LoadIris()
Xs, Ys = l_iris.get_all()

if True:
    l_mnist = LoadMnist()
    Xs, Ys = l_mnist.get_all()
    X = []
    for xs_ in Xs:
        X.append([int(x) for x2 in xs_ for x in x2])
    Xs = X
print("loaded")

X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.3, random_state=0)

codings, reconstructed = reconstruct_data(X_train, X_test, int(len(X_train[0])/2))
print("got reconstructed")
#calc_differences(X_test, reconstructed)

plt.plot(X_test, c='g')
plt.plot(reconstructed, c='b')
plt.plot(codings, c='r')
custom_lines = [Line2D([0], [0], color='g', lw=4), Line2D([0], [0], color='b', lw=4)]
plt.legend(custom_lines, ['original', 'reconstructed'])
plt.show()