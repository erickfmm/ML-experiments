from keras.datasets import mnist
from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadMnist",]

class LoadMnist(ILoadSupervised):
    def __init__(self):
        (self.XTrain,self.YTrain),(self.XTest,self.YTest)=mnist.load_data()

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        return None