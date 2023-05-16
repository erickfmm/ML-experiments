from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join

__all__ = ["LoadVehicle"]


class LoadVehicle(ILoadSupervised):
    def __init__(self, folder_path="train_data/Folder_Basic/vehicle/"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folder_path
        self.headers = ["COMPACTNESS", "CIRCULARITY",
                        "DISTANCE CIRCULARITY", "RADIUS RATIO",
                        "PR.AXIS ASPECT RATIO", "MAX.LENGTH ASPECT RATIO",
                        "SCATTER RATIO", "ELONGATEDNESS", "RECTANGULARITY",
                        "MAX.LENGTH RECTANGULARITY",
                        "SCALED VARIANCE ALONG MAJOR AXIS",
                        "SCALED VARIANCE ALONG MINOR AXIS ",
                        "SCALED RADIUS OF GYRATION",
                        "SKEWNESS ABOUT MAJOR AXIS",
                        "SKEWNESS ABOUT MINOR AXIS",
                        "KURTOSIS ABOUT MINOR AXIS",
                        "KURTOSIS ABOUT MAJOR AXIS",
                        "HOLLOWS RATIO"]
        self.classes = ["OPEL", "SAAB", "BUS", "VAN"]
    
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
        data_vs = []
        file_a = open(join(self.folder_path, 'xab.dat'))
        file_b = open(join(self.folder_path, 'xaa.dat'))
        file_c = open(join(self.folder_path, 'xac.dat'))
        file_d = open(join(self.folder_path, 'xad.dat'))
        file_e = open(join(self.folder_path, 'xae.dat'))
        file_f = open(join(self.folder_path, 'xaf.dat'))
        file_g = open(join(self.folder_path, 'xag.dat'))
        file_h = open(join(self.folder_path, 'xah.dat'))
        file_i = open(join(self.folder_path, 'xai.dat'))
        data_vs.append(csv.reader(file_a, delimiter=' '))
        data_vs.append(csv.reader(file_b, delimiter=' '))
        data_vs.append(csv.reader(file_c, delimiter=' '))
        data_vs.append(csv.reader(file_d, delimiter=' '))
        data_vs.append(csv.reader(file_e, delimiter=' '))
        data_vs.append(csv.reader(file_f, delimiter=' '))
        data_vs.append(csv.reader(file_g, delimiter=' '))
        data_vs.append(csv.reader(file_h, delimiter=' '))
        data_vs.append(csv.reader(file_i, delimiter=' '))
        i = 0
        for csv_reader in data_vs:
            for row in csv_reader:
                x = []
                y = None
                i_field = 0
                for field in row:
                    if i_field < len(row) - 1:
                        x.append(int(field))
                    else:
                        y = field
                    i_field += 1
                i += 1
                yield x, y
        file_a.close()
        file_b.close()
        file_c.close()
        file_d.close()
        file_e.close()
        file_f.close()
        file_g.close()
        file_h.close()
        file_i.close()
