from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join

__all__ = ["LoadGlass",]

class LoadGlass(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/glass/"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folderpath
        self.headers = ["RI: refractive index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum", "Si: Silicon", "K: Potassium", "Ca: Calcium", "Ba: Barium", "Fe: Iron"]
        self.classes = [None, "building_windows_float_processed", "building_windows_non_float_processed", "vehicle_windows_float_processed", "vehicle_windows_non_float_processed (none in this database)", "containers", "tableware", "headlamps"]

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
        for x, y in self.get_all_yielded():
            Xs.append(x)
            Ys.append(y)
        return Xs, Ys

    def get_all_yielded(self):
        i = 0
        with open(join(self.folder_path, 'glass.data')) as data_glass_file:
            data_glass_csv = csv.reader(data_glass_file)
            for el in data_glass_csv:
                iField = 0
                x = []
                for field in el:
                    if iField > 0 and iField < len(el)-2:
                        x.append(float(field))
                    elif iField > len(el) - 2:
                        y = int(field)
                    iField += 1
                yield x, y
                i += 1
