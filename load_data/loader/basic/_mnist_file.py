import mnist
from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadMnist"]

fashion_path = 'train_data/Folder_Images_Supervised/fashionmnist'
kuzushiji_path = 'train_data/Folder_Images_Supervised/kuzushiji'


class LoadMnist(ILoadSupervised):
    # fashion: mnist_path='train_data\\Folder_Images_Supervised\\fashionmnist'
    def __init__(self, mnist_path='train_data/Folder_Images_Supervised/mnist'):
        self.TYPE = SupervisedType.Classification
        self.mnist_data = mnist.parse_idx(mnist_path)
        self.XTrain, self.YTrain = self.mnist_data.load_training()
        self.XTest, self.YTest = self.mnist_data.load_testing()
        # print(mnist_data.display(images[index]))
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
        xs = []
        ys = []
        for i_train in range(len(self.XTrain)):
            xs.append(self.XTrain[i_train])
            ys.append(self.YTrain[i_train])
        for i_test in range(len(self.XTest)):
            xs.append(self.XTest[i_test])
            ys.append(self.YTest[i_test])
        return xs, ys


# call gen_image(x[0]).show()
def gen_image(arr):
    from matplotlib import pyplot as plt
    import numpy as np
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, cmap='gray', interpolation='nearest')
    return plt
