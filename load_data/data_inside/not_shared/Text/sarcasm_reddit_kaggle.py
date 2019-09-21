from load_data.ILoadSupervised import ILoadSupervised
import csv
from os.path import join

__all__ = ["LoadSarcasmRedditKaggle",]



class LoadSarcasmRedditKaggle(ILoadSupervised):
    def __init__(self, folderpath="train_data\\not_shared\\Folder_FromKaggle\\sarcasm"):
        self.folderpath = folderpath
        self.classes = [0, 1]
        self.headers = []

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.read_train()
    
    def read_train(self):
        self.X = []
        self.Y = []
        self.Metadata = []
        self.MetadataHeaders = ["author", "subreddit", "score", "ups", "downs","date","created_utc","parent_comment"]
        with open(join(self.folderpath, "train-balanced-sarcasm.csv"), "r") as fileobj:
            file_reader = csv.DictReader(fileobj, delimiter=',')
            for row in file_reader:
                self.Y.append(int(row["label"]))
                self.X.append(row["comment"])
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
        return self.X, self.Y
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
