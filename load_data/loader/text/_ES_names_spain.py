from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join
import csv

__all__ = ["LoadNamesSpain"]


class LoadNamesSpain(ILoadUnsupervised):
    """docstring for LoadNamesSpain"""
    def __init__(self,
                 folder_path: str = "train_data/Folder_NLPEspañol/spanish-names",
                 females: bool = True,
                 males: bool = True):
        # super(LoadNamesSpain, self).__init__()
        self.folder_path: str = folder_path
        self.females: bool = females
        self.males: bool = males

    def get_headers(self):
        return None

    def get_all(self):
        words = []
        for name, freq, age in self.get_all_yielded():
            words.append([name, freq, age])
        return words
    
    def get_all_yielded(self):
        if self.females:
            for name, freq, age in LoadNamesSpain.read_file_csv(join(self.folder_path, "female_names.csv")):
                yield name, freq, age
        if self.males:
            for name, freq, age in LoadNamesSpain.read_file_csv(join(self.folder_path, "male_names.csv")):
                yield name, freq, age

    @staticmethod
    def read_file_csv(filepath: str):
        with open(filepath, "r", encoding="utf-8") as file_obj:
            file_reader = csv.DictReader(file_obj, delimiter=',')
            for row in file_reader:
                yield row["name"].lower(), int(row["frequency"]), float(row["mean_age"])
