from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv

__all__ = ["LoadMnistCsv",]

class LoadMnistCsv(ILoadSupervised):
    def __init__(self, 
        trainfile='train_data/Folder_Images_Supervised/sign-language-mnist/sign_mnist_test.csv',
        testfile='train_data/Folder_Images_Supervised/sign-language-mnist/sign_mnist_train.csv'):
        self.TYPE = SupervisedType.Classification
        self.XTrain, self.YTrain = self.load_file(trainfile)
        self.XTest, self.YTest = self.load_file(testfile)
        self.trainfile = trainfile
        self.testfile = testfile
    
    def get_classes(self):
        self.classes = [str(i) for i in list(set(self.YTrain))]
        return self.classes
    
    def get_headers(self):
        self.headers = [str(i) for i in range(len(self.XTest[0]))]
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
    
    @staticmethod
    def load_file(filepath):
        Xs = []
        Ys = []
        with open(filepath, "r") as fileobj:
            reader = csv.reader(fileobj)
            irow = -1
            for row in reader:
                if irow >= 0:
                    Xs.append([])
                    for icol in range(len(row)):
                        if icol == 0:
                            Ys.append(int(row[icol]))
                        else:
                            Xs[irow].append(int(row[icol]))    
                irow += 1
        return Xs, Ys