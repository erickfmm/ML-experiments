from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
from os.path import join
import opendatasets as od

__all__ = ["LoadRedditOrTwitterSentiment"]


class LoadRedditOrTwitterSentiment(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/NLP_ENG_Sentiment/twitter-and-reddit-sentimental-analysis-dataset", source="twitter"):
        self.TYPE = SupervisedType.Unknown
        self.file_path = join(folder_path, "Reddit_Data.csv") if source == "reddit" else join(folder_path, "Twitter_Data.csv")
        self.source = source
    
    def get_X_Y(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(file_obj, delimiter=",")
        xs = []
        ys = []
        for row in reader:
            if len(row) == 2 and row[1] != 'category':
                xs.append(row[0])
                y = int(row[1])
                if y == -1:
                    y = 0
                elif y == 0:
                    y = 0.5
                ys.append(y)
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return [0,1,-1]
    
    def get_headers(self):
        return None

    def download(self, folder_path="data/train_data/NLP_ENG_Sentiment"):
        od.download("https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset", folder_path)
