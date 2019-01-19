from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadGlass",]

class LoadGlass(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/glass/"):
        self.folder_path = folderpath

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        Xs = []
        Ys = []
        i = 0
        with open(join(self.folder_path, 'glass.data')) as data_glass_file:
            data_glass_csv = csv.reader(data_glass_file)
            for el in data_glass_csv:
                iField = 0
                Xs.append([])
                for field in el:
                    if iField > 0 and iField < len(el)-2:
                        Xs[i].append(float(field))
                    elif iField > len(el) - 2:
                        Ys.append(int(field))
                    iField += 1
                i += 1
        return Xs, Ys
