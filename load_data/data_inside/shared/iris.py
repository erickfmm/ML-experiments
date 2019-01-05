from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadIris",]

class LoadIris(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/iris/"):
        self.folder_path = folderpath

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        data_iris_csv = csv.reader(open(join(self.folder_path,'iris.data')))
        Xs = []
        Ys = []
        i = 0
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
