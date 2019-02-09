from mnist import MNIST
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadMnist",]

fashion_path = 'train_data\\not_shared\\Folder_FromKaggle\\fashionmnist'
kuzushiji_path = 'train_data\\not_shared\\Folder_FromKaggle\\kuzushiji'

class LoadMnist(ILoadSupervised):
    #fashion: mnist_path='train_data\\not_shared\\Folder_FromKaggle\\fashionmnist'
    def __init__(self, mnist_path='train_data\\not_shared\\mnist'):
        self.TYPE = SupervisedType.Classification
        self.mndata = MNIST(mnist_path)
        self.XTrain, self.YTrain = self.mndata.load_training()
        self.XTest, self.YTest = self.mndata.load_testing()
        #print(mndata.display(images[index]))
        self.headers = [str(i) for i in range(len(self.XTest[0]))]
        self.classes = [str(i) for i in list(set(self.YTrain))]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

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



#call gen_image(x[0]).show()
def gen_image(arr):
    from matplotlib import pyplot as plt
    import numpy as np
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, cmap='gray', interpolation='nearest')
    return plt