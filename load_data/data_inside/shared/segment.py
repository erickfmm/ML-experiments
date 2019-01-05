from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadSegment",]

class LoadSegment(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/segment/"):
        self.folder_path = folderpath

    def get_default(self):
        return self.get_splited()

    def get_splited(self):
        data_segment_csv = csv.reader(open(join(self.folder_path, 'segmentation.data')))
        data_segment_test_csv = csv.reader(open(join(self.folder_path, 'segmentation.test')))
        X_train = []
        Y_train = []
        i = 0
        for row in data_segment_csv:
            if i >= 5:
                X_train.append([])
                iField = 0
                for field in row:
                    if iField == 0:
                        Y_train.append(field)
                    else:
                        X_train[i-5].append(float(field))
                    iField += 1
                #print(row)
            i += 1

        X_test = []
        Y_test = []
        i2 = 0
        for row in data_segment_test_csv:
            if i2 >= 5:
                X_test.append([])
                iField = 0
                for field in row:
                    if iField == 0:
                        Y_test.append(field)
                    else:
                        X_test[i-5].append(float(field))
                    iField += 1
                #print(row)
                i += 1
            i2 += 1
        return (X_train, Y_train), (X_test, Y_test)
    
    def get_all(self):
        data_segment_csv = csv.reader(open(join(self.folder_path, 'segmentation.data')))
        data_segment_test_csv = csv.reader(open(join(self.folder_path, 'segmentation.test')))
        Xs = []
        Ys = []
        i = 0
        for row in data_segment_csv:
            if i >= 5:
                Xs.append([])
                iField = 0
                for field in row:
                    if iField == 0:
                        Ys.append(field)
                    else:
                        Xs[i-5].append(float(field))
                    iField += 1
            i += 1

        i2 = 0
        for row in data_segment_test_csv:
            if i2 >= 5:
                Xs.append([])
                iField = 0
                for field in row:
                    if iField == 0:
                        Ys.append(field)
                    else:
                        Xs[i-5].append(float(field))
                    iField += 1
                i += 1
            i2 += 1
        return Xs, Ys
