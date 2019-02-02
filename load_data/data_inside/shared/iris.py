from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join

__all__ = ["LoadIris",]

class LoadIris(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/iris/"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folderpath
        self.headers = ["sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm"]
        self.classes = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
    
    def get_all(self):
        Xs = []
        Ys = []
        i = 0
        with open(join(self.folder_path,'iris.data')) as data_iris_file:
            data_iris_csv = csv.reader(data_iris_file)
            for row in data_iris_csv:
                if len(row) > 0:
                    Xs.append([])
                    iField = 0
                    for field in row:
                        if iField < len(row)-1:
                            Xs[i].append(float(field))
                        else:
                            Ys.append(field)
                        iField += 1
                i += 1
        return Xs, Ys
