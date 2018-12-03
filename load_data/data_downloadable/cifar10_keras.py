from keras.datasets import cifar10
from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadCifar10",]

class LoadCifar10(ILoadSupervised):
    def __init__(self):
        (self.XTrain,self.YTrain),(self.XTest,self.YTest)=cifar10.load_data()

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        return None