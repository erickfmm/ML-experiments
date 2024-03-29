from keras.datasets import cifar10
from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadCifar10"]


class LoadCifar10(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification

    def get_classes(self):
        return []
    
    def get_headers(self):
        return ["pixels"]

    @staticmethod
    def get_default():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return x_train, y_train

    @staticmethod
    def get_splited():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return (x_train, y_train), (x_test, y_test)
    
    def get_X_Y(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return np.append(x_train, x_test, 0), np.append(y_train, y_test, 0)
