import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
######################################################

from mlexperiments.load_data.loader.downloadable.mnist_keras import LoadMnist
from mlexperiments.load_data.loader.downloadable.cifar10_keras import LoadCifar10
from mlexperiments.utils.keras_persistence.all_inside import save

import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.utils import to_categorical

#dataset_to_use = "cifar10"
dataset_to_use = "mnist"
# dataset_to_use = "fashion"

if dataset_to_use == "cifar10":
    l_cifar10 = LoadCifar10()
    (Xtrain, Ytrain), (Xtest, Ytest) = l_cifar10.get_splited()
    Ytrain = Ytrain.reshape(len(Ytrain))
    Ytest = Ytest.reshape(len(Ytest))

if dataset_to_use == "mnist":
    l_mnist = LoadMnist()
    (Xtrain, Ytrain), (Xtest, Ytest) = l_mnist.get_splited()

n_classes = len(set(Ytrain))
print("number of classes are: ", n_classes)

n_train=len(Xtrain)  # 50000 datos de entrenamiento
n_test=len(Xtest)  # 10000 datos de test
print("n train: ", n_train)
print("n test: ", n_test)

# Modificando X
# ------------
# this shapes are for mnist
if dataset_to_use in ["mnist", "fashion"]:
    Xtrain = Xtrain.reshape(n_train,28,28,1)
    Xtest = Xtest.reshape(n_test,28,28,1)
    shape_for_nn = (28,28,1)
# / -------------

# ------------
# this shapes are for cifar10
if dataset_to_use == "cifar10":
    Xtrain = Xtrain.reshape(n_train,32,32,3)
    Xtest = Xtest.reshape(n_test,32,32,3)
    shape_for_nn = (32,32,3)
# / -------------
Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain /= 255
Xtest /= 255

Ytrain = to_categorical(Ytrain, n_classes)
Ytest = to_categorical(Ytest, n_classes)


print("creating the model...")
model = Sequential()
model.add(Convolution2D(12,(5,5),padding='same',activation='relu',input_shape=shape_for_nn))
model.add(MaxPooling2D())
model.add(Convolution2D(18,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(24,(3,3),padding='same',activation='relu'))
model.add(Flatten())  # Reduciendo dimensionalidad
model.add(Dropout(0.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(n_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("start to train")
# Entrenamiento
# model.fit(Xtrain,Ytrain,epochs=15,batch_size=32,verbose=1,shuffle=True)
model.fit(Xtrain,Ytrain,epochs=10,batch_size=32,verbose=1,shuffle=True)


score=model.evaluate(Xtest,Ytest,verbose=1)
print("\n Loss: %.3f \t Accuracy: %.3f"%(score[0],score[1]))

save(model, os.path.join("data", "created_models", "mnist_convolve.model"))