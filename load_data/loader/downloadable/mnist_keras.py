from keras.datasets import mnist
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import numpy as np

__all__ = ["LoadMnist",]

class LoadMnist(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        #self.headers = [str(i) for i in range(len(self.XTest[0]))]
        self.classes = [str(i) for i in range(10)]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return [] #self.headers

    def get_splited(self):
        (XTrain,YTrain),(XTest,YTest)=mnist.load_data()
        return (XTrain, YTrain), (XTest, YTest)
    
    def get_all(self):
        (XTrain,YTrain),(XTest,YTest)=mnist.load_data()
        return np.append(XTrain, XTest, 0), np.append(YTrain, YTest, 0)