from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join

__all__ = ["LoadGlass"]


class LoadGlass(ILoadSupervised):
    def __init__(self, folder_path="train_data/Folder_Basic/glass/"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folder_path
        self.headers = ["RI: refractive index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum", "Si: Silicon",
                        "K: Potassium", "Ca: Calcium", "Ba: Barium", "Fe: Iron"]
        self.classes = [None, "building_windows_float_processed", "building_windows_non_float_processed",
                        "vehicle_windows_float_processed",
                        "vehicle_windows_non_float_processed (none in this database)",
                        "containers", "tableware", "headlamps"]

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
    
    def get_all(self):
        xs = []
        ys = []
        for x, y in self.get_all_yielded():
            xs.append(x)
            ys.append(y)
        return xs, ys

    def get_all_yielded(self):
        i = 0
        with open(join(self.folder_path, 'glass.data')) as data_glass_file:
            data_glass_csv = csv.reader(data_glass_file)
            for el in data_glass_csv:
                i_field = 0
                x = []
                for field in el:
                    if 0 < i_field < len(el)-2:
                        x.append(float(field))
                    elif i_field > len(el) - 2:
                        y = int(field)
                    i_field += 1
                yield x, y
                i += 1
