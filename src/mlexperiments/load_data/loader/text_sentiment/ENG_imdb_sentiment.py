from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import csv
import opendatasets as od

__all__ = ["LoadImdbSentiment"]


class LoadImdbSentiment(ILoadSupervised):
    def __init__(self, file_path="data/train_data/NLP_ENG_Sentiment/imdb-movie-ratings-sentiment-analysis/movie.csv"):
        self.TYPE = SupervisedType.Unknown
        self.file_path = file_path
    
    def get_X_Y(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(file_obj, delimiter=",")
        xs = []
        ys = []
        for row in reader:
            if row[1] != 'label':
                xs.append(row[0])
                ys.append(int(row[1]))
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return None
    
    def get_headers(self):
        return None

    def download(self, folder_path="data/train_data/NLP_ENG_Sentiment"):
        od.download("https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis", folder_path)
