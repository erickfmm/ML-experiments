from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadSentiment140"]


class LoadSentiment140(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/NLP_ENG_Sentiment/sentiment140/training.1600000.processed.noemoticon.csv"):
        self.file_path = folder_path

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(file_obj, delimiter=",")
        xs = []
        ys = []
        for row in reader:
            xs.append(row[5])
            y = int(row[0]) if row[0] == "0" else 1
            ys.append(y)
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return [0,1]
    
    def get_headers(self):
        return None
