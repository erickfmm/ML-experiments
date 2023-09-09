from keras.datasets import fashion_mnist
from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadFashionMnist"]


class LoadFashionMnist(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
    
    def get_classes(self):
        return [str(i) for i in range(10)]
    
    def get_headers(self):
        return ["pixels"]

    def get_default(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return x_train, y_train

    def get_splited(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return (x_train, y_train), (x_test, y_test)
    
    def get_X_Y(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return np.append(x_train, x_test, 0), np.append(y_train, y_test, 0)
