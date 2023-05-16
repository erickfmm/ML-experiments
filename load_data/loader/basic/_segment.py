from load_data.ILoadSupervised import ILoadSupervised, SupervisedType, ISplitted
import csv
from os.path import join

__all__ = ["LoadSegment"]


class LoadSegment(ILoadSupervised, ISplitted):
    def __init__(self, folder_path="train_data/Folder_Basic/segment/"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folder_path
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
        x_train = []
        y_train = []
        i = 0
        for row in data_segment_csv:
            if i >= 5:
                x_train.append([])
                i_field = 0
                for field in row:
                    if i_field == 0:
                        y_train.append(field)
                    else:
                        x_train[i-5].append(float(field))
                    i_field += 1
                # print(row)
            i += 1
        segment_data_file.close()
        return x_train, y_train

    def get_test(self):
        segment_test_file = open(join(self.folder_path, 'segmentation.test'))
        data_segment_test_csv = csv.reader(segment_test_file)

        x_test = []
        y_test = []
        i2 = 0
        for row in data_segment_test_csv:
            if i2 >= 5:
                x_test.append([])
                i_field = 0
                i = 0  # TODO: test
                for field in row:
                    if i_field == 0:
                        y_test.append(field)
                    else:
                        x_test[i-5].append(float(field))
                    i_field += 1
                # print(row)
                i += 1
            i2 += 1
        segment_test_file.close()
        return x_test, y_test

    def get_splited(self):
        x_train, y_train = self.get_train()
        x_test, y_test = self.get_test()
        return (x_train, y_train), (x_test, y_test)
    
    def get_train_yielded(self):
        x_train, y_train = self.get_train()
        for i in range(len(x_train)):
            yield x_train[i], y_train[i]
    
    def get_test_yielded(self):
        x_test, y_test = self.get_test()
        for i in range(len(x_test)):
            yield x_test[i], y_test[i]

    def get_all(self):
        segment_data_file = open(join(self.folder_path, 'segmentation.data'))
        data_segment_csv = csv.reader(segment_data_file)
        segment_test_file = open(join(self.folder_path, 'segmentation.test'))
        data_segment_test_csv = csv.reader(segment_test_file)
        xs = []
        ys = []
        i = 0
        for row in data_segment_csv:
            if i >= 5:
                xs.append([])
                i_field = 0
                for field in row:
                    if i_field == 0:
                        ys.append(field)
                    else:
                        xs[i-5].append(float(field))
                    i_field += 1
            i += 1

        i2 = 0
        for row in data_segment_test_csv:
            if i2 >= 5:
                xs.append([])
                i_field = 0
                for field in row:
                    if i_field == 0:
                        ys.append(field)
                    else:
                        xs[i-5].append(float(field))
                    i_field += 1
                i += 1
            i2 += 1
        segment_data_file.close()
        segment_test_file.close()
        return xs, ys

    def get_all_yielded(self):
        xs, ys = self.get_all()
        for i in range(len(xs)):
            yield xs[i], ys[i]
