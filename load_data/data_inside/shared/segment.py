from load_data.ILoadSupervised import ILoadSupervised, SupervisedType, ISplitted
import csv
from os.path import join

__all__ = ["LoadSegment",]

class LoadSegment(ILoadSupervised, ISplitted):
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

    def get_train(self):
        segment_data_file = open(join(self.folder_path, 'segmentation.data'))
        data_segment_csv = csv.reader(segment_data_file)
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
        segment_data_file.close()
        return (X_train, Y_train)

    def get_test(self):
        segment_test_file = open(join(self.folder_path, 'segmentation.test'))
        data_segment_test_csv = csv.reader(segment_test_file)

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
        segment_test_file.close()
        return X_test, Y_test

    def get_splited(self):
        X_train, Y_train = self.get_train()
        X_test, Y_test = self.get_test()
        return (X_train, Y_train), (X_test, Y_test)
    
    def get_train_yielded(self):
        X_train, Y_train = self.get_train()
        for i in range(len(X_train)):
            yield X_train[i], Y_train[i]
    
    def get_test_yielded(self):
        X_test, Y_test = self.get_test()
        for i in range(len(X_test)):
            yield X_test[i], Y_test[i]

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

    def get_all_yielded(self):
        xs, ys = self.get_all()
        for i in range(len(xs)):
            yield xs[i], ys[i]