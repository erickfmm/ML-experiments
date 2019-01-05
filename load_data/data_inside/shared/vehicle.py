from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadVehicle",]

class LoadVehicle(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/vehicle/"):
        self.folder_path = folderpath

    def get_default(self):
        return None

    def get_splited(self):
        return None
    
    def get_all(self):
        data_vs = []
        data_vs.append(csv.reader(open(join(self.folder_path, 'xaa.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xab.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xac.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xad.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xae.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xaf.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xag.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xah.dat')), delimiter=' '))
        data_vs.append(csv.reader(open(join(self.folder_path, 'xai.dat')), delimiter=' '))
        Xs = []
        Ys = []
        i = 0
        for csv_reader in data_vs:
            for row in csv_reader:
                Xs.append([])
                iField = 0
                for field in row:
                    if iField < len(row) -1:
                        Xs[i].append(int(field))
                    else:
                        Ys.append(field)
                    iField += 1
                i += 1
        return Xs, Ys
