from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join

__all__ = ["LoadSteam"]


class LoadSteam(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/NLP_ENG_Sentiment/sentiment-analysis-for-steam-reviews/train.csv"):
        self.TYPE = SupervisedType.Unknown
        self.file_path = folder_path
    
    def get_X_Y(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(file_obj, delimiter=",")
        xs = []
        ys = []
        for row in reader:
            xs.append(row[3])
            ys.append(row[4])
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return [0,1]
    
    def get_headers(self):
        return None
