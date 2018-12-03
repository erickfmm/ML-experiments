from mnist import MNIST
from load_data.ILoadSupervised import ILoadSupervised

class LoadMnist(ILoadSupervised):
    def __init__(self):
        self.mndata = MNIST('train_data\\mnist')
        self.XTrain, self.YTrain = self.mndata.load_training()
        self.XTest, self.YTest = self.mndata.load_testing()
        #print(mndata.display(images[index]))

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        return None