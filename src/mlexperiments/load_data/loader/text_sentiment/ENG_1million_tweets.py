from mlexperiments.load_data.ILoadSupervised import ILoadSupervised
import csv
import opendatasets as od

__all__ = ["Load1MTweets"]


class Load1MTweets(ILoadSupervised):
    def __init__(self, file_path="data/train_data/NLP_ENG_Sentiment/sentiment-dataset-with-1-million-tweets/dataset.csv", lang=None):
        self.file_path = file_path
        self.lang = lang
        self.Metadata = []
    
    def get_X_Y(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.DictReader(file_obj, delimiter=",")
        xs = []
        ys = []
        for row in reader:
            if self.lang is not None and self.lang != row["Language"]:
                continue
            if row["Label"] in ['negative', 'positive']:
                xs.append(row["Text"])
                y = 0 if row["Label"] == 'negative' else 1
                ys.append(y)
                self.Metadata.append(row["Language"])
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return None
    
    def get_headers(self):
        return None

    def download(self, folder_path="data/train_data/NLP_ENG_Sentiment"):
        od.download("https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets", folder_path)
