from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadSarcasmRedditKaggle"]


class LoadSarcasmRedditKaggle(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/NLP_ENG_Dialogs/sarcasm"):
        self.folder_path = folder_path
        self.classes = [0, 1]
        self.headers = []
        self.Metadata = []
        self.MetadataHeaders = ["author", "subreddit", "score", "ups", "downs", "date", "created_utc", "parent_comment"]

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        return self.read_train()
    
    def read_train(self):
        xs = []
        ys = []
        with open(join(self.folder_path, "train-balanced-sarcasm.csv"), "r") as fileobj:
            file_reader = csv.DictReader(fileobj, delimiter=',')
            for row in file_reader:
                ys.append(int(row["label"]))
                xs.append(row["comment"])
                self.Metadata.append([
                    row["author"],
                    row["subreddit"],
                    int(row["score"]),
                    int(row["ups"]),
                    int(row["downs"]),
                    row["date"],
                    row["created_utc"],
                    row["parent_comment"]
                ])
        return xs, ys
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
