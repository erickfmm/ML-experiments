import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

from mlexperiments.unsupervised.autoencoder_tf import reconstruct_data

#from load_data.loader.basic.iris import LoadIris
from mlexperiments.load_data.loader.downloadable.iris_sklearn import LoadIris
from mlexperiments.load_data.loader.downloadable.mnist_keras import LoadMnist


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
Xs, Ys = l_iris.get_X_Y()

if True:
    l_mnist = LoadMnist()
    Xs, Ys = l_mnist.get_X_Y()
    X = []
    for xs_ in Xs:
        X.append(np.asarray([x for x2 in xs_ for x in x2]))
    Xs = np.asarray(X)
print("loaded")

X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=0.3, random_state=0)

codings, reconstructed = reconstruct_data(X_train, X_test, int(len(X_train[0])/(4*3)))
print("got reconstructed")
#calc_differences(X_test, reconstructed)
#X_test_2d = [[x for x in x2] for xs_ in X_test for x2 in xs_]
#reconstructed_2d = [[x for x in x2] for xs_ in reconstructed for x2 in xs_]
rows, columns = 2,2
fig = plt.figure(figsize=(rows, columns))
fig.add_subplot(rows, columns, 1) 
plt.axis('off') 
plt.imshow(np.asarray(X_test[100]).reshape((28,28)), 'gray')#, '*-', c='g',)
#plt.show()
fig.add_subplot(rows, columns, 2) 
plt.axis('off') 
plt.imshow(np.asarray(reconstructed[100]).reshape((28,28)), 'gray')#,'.-', c='b')

fig.add_subplot(rows, columns, 3) 
plt.axis('off') 
plt.imshow(np.asarray(X_test[50]).reshape((28,28)), 'gray')#,'.-', c='b')

fig.add_subplot(rows, columns, 4) 
plt.axis('off') 
plt.imshow(np.asarray(reconstructed[50]).reshape((28,28)), 'gray')#,'.-', c='b')

plt.show()
#codings_2d = [[x for x in x2] for xs_ in codings for x2 in xs_]
#plt.plot(codings[:5],'.', c='r')
#custom_lines = [Line2D([0], [0], color='g', lw=4), Line2D([0], [0], color='b', lw=4)]
#plt.legend(custom_lines, ['original', 'reconstructed'])
#plt.show()