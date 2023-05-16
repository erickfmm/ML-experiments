from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv

__all__ = ["LoadMnistCsv"]


class LoadMnistCsv(ILoadSupervised):
    def __init__(self,
                 train_file='train_data/Folder_Images_Supervised/sign-language-mnist/sign_mnist_test.csv',
                 test_file='train_data/Folder_Images_Supervised/sign-language-mnist/sign_mnist_train.csv'):
        self.TYPE = SupervisedType.Classification
        self.XTrain, self.YTrain = self.load_file(train_file)
        self.XTest, self.YTest = self.load_file(test_file)
        self.classes = [str(i) for i in list(set(self.YTrain))]
        self.headers = [str(i) for i in range(len(self.XTest[0]))]
    
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
    
    @staticmethod
    def load_file(filepath):
        xs = []
        ys = []
        with open(filepath, "r") as file_obj:
            reader = csv.reader(file_obj)
            i_row = -1
            for row in reader:
                if i_row >= 0:
                    xs.append([])
                    for i_col in range(len(row)):
                        if i_col == 0:
                            ys.append(int(row[i_col]))
                        else:
                            xs[i_row].append(int(row[i_col]))
                i_row += 1
        return xs, ys
