from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join
import opendatasets as od

__all__ = ["LoadSpanishNews"]


class LoadSpanishNews(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/NLP_ESP/noticias-laraznpblico"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folder_path
    
    def get_X_Y(self):
        X = []
        Y = []
        for cuerpo, titular in self.get_X_Y_yielded():
            X.append(cuerpo)
            Y.append(titular)
        return X, Y
    
    
    def get_X_Y_yielded(self):
        with open(join(self.folder_path, "data_larazon_publico_v2.csv"), "r", encoding="utf-8") as fileobj:
            file_reader = csv.DictReader(fileobj, delimiter=',')
            for row in file_reader:
                yield row["cuerpo"], row["titular"]


    def get_classes(self):
        return None
    
    def get_headers(self):
        return None

    def download(self, folder_path="data/train_data/NLP_ESP"):
        od.download("https://www.kaggle.com/datasets/josemamuiz/noticias-laraznpblico", folder_path)
