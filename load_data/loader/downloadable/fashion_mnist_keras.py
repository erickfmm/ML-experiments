from keras.datasets import fashion_mnist
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadFashionMnist"]


class LoadFashionMnist(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self.headers = ["pixels"]
        self.classes = [str(i) for i in range(10)]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return x_train, y_train

    def get_splited(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return (x_train, y_train), (x_test, y_test)
    
    def get_all(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        return np.append(x_train, x_test, 0), np.append(y_train, y_test, 0)
