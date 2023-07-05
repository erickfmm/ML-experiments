from load_data.ILoadSupervised import ILoadSupervised
import csv

__all__ = ["Load1MTweets"]


class Load1MTweets(ILoadSupervised):
    def __init__(self, file_path="data/train_data/NLP_ENG_Sentiment/sentiment-dataset-with-1-million-tweets/dataset.csv", lang=None):
        self.file_path = file_path
        self.lang = lang
        self.Metadata = []

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.DictReader(file_obj, delimiter=",")
        xs = []
        ys = []
        for row in reader:
            if self.lang is not None and self.lang != row["Language"]:
                continue
            xs.append(row["Text"])
            ys.append(row["Label"])
            self.Metadata.append(row["Language"])
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return None
    
    def get_headers(self):
        return None
