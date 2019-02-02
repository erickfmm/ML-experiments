from keras.datasets import mnist
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadMnist",]

class LoadMnist(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        (self.XTrain,self.YTrain),(self.XTest,self.YTest)=mnist.load_data()
        self.headers = [str(i) for i in range(len(self.XTest[0]))]
        self.classes = [str(i) for i in range(10)]
    
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