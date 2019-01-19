from mnist import MNIST
from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadMnist",]

class LoadMnist(ILoadSupervised):
    #fashion: mnist_path='train_data\\not_shared\\Folder_FromKaggle\\fashionmnist'
    def __init__(self, mnist_path='train_data\\not_shared\\mnist'):
        self.mndata = MNIST(mnist_path)
        self.XTrain, self.YTrain = self.mndata.load_training()
        self.XTest, self.YTest = self.mndata.load_testing()
        #print(mndata.display(images[index]))

    def get_default(self):
        return self.XTrain, self.YTrain

    def get_splited(self):
        return (self.XTrain, self.YTrain), (self.XTest, self.YTest)
    
    def get_all(self):
        Xs = []
        Ys = []
        for i_train in range(len(self.XTrain)):
            Xs.append(self.XTrain[i_train])
            Ys.append(self.YTrain[i_train])
        for i_test in range(len(self.XTest)):
            Xs.append(self.XTest[i_test])
            Ys.append(self.YTest[i_test])
        return Xs, Ys