from keras.datasets import cifar10
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadCifar10",]

class LoadCifar10(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        (self.XTrain,self.YTrain),(self.XTest,self.YTest)=cifar10.load_data()
        self.headers = ["pixels"]
        self.classes = []
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        return np.append(self.XTrain, self.XTest, 0), np.append(self.YTrain, self.YTest, 0)