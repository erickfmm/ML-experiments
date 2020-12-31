from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join
import csv

#"name","frequency","mean_age"

class LoadNamesSpain(ILoadUnsupervised):
    """docstring for LoadNamesSpain"""
    def __init__(self, 
        datapath: str ="train_data/Folder_NLPEspañol/spanish-names",
        females: bool = True,
        males: bool = True):
        #super(LoadNamesSpain, self).__init__()
        self.datapath = datapath
        self.females = females
        self.males = males

    def get_headers(self):
        return None

    def get_all(self):
        words = []
        for name, freq, age in self.get_all_yielded():
            words.append([name, freq, age])
        return words
    
    def get_all_yielded(self):
        if self.females:
            for name, freq, age in LoadNamesSpain.read_file_csv(join(self.datapath, "female_names.csv")):
                yield name, freq, age
        if self.males:
            for name, freq, age in LoadNamesSpain.read_file_csv(join(self.datapath, "female_names.csv")):
                yield name, freq, age

    @staticmethod
    def read_file_csv(filepath: str):
        fobj = open(filepath, "r", encoding="utf-8")
        file_reader = csv.DictReader(fobj, delimiter=',')
        for row in file_reader:
            yield (row["name"].lower(), int(row["frequency"]), float(row["mean_age"]))