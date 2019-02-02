from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join

__all__ = ["LoadSegment",]

class LoadSegment(ILoadSupervised):
    def __init__(self, folderpath="train_data/shared/segment/"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folderpath
        self.headers = ["REGION-CENTROID-ROW", "REGION-PIXEL-COUNT",
        "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2",
        "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN", "HEDGE-SD",
        "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN",
        "RAWGREEN-MEAN", "EXRED-MEAN", "EXBLUE-MEAN", "EXGREEN-MEAN",
        "VALUE-MEAN", "SATURATION-MEAN", "HUE-MEAN"]
        self.classes = ["brickface", "sky", "foliage", "cement", "window", "path", "grass"]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        return self.get_splited()

    def get_splited(self):
        segment_data_file = open(join(self.folder_path, 'segmentation.data'))
        data_segment_csv = csv.reader(segment_data_file)
        segment_test_file = open(join(self.folder_path, 'segmentation.test'))
        data_segment_test_csv = csv.reader(segment_test_file)
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
        segment_data_file.close()
        segment_test_file.close()
        return (X_train, Y_train), (X_test, Y_test)
    
    def get_all(self):
        segment_data_file = open(join(self.folder_path, 'segmentation.data'))
        data_segment_csv = csv.reader(segment_data_file)
        segment_test_file = open(join(self.folder_path, 'segmentation.test'))
        data_segment_test_csv = csv.reader(segment_test_file)
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
        segment_data_file.close()
        segment_test_file.close()
        return Xs, Ys
