from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join
import csv

class LoadInternationalStopWords(ILoadUnsupervised):
    def __init__(self, 
        datafile="train_data/Folder_NLPEspañol/internationalstopwords/International-Stop-Words.csv",
        lang="Spanish"):
        self.datafile = datafile
        self.lang = lang if lang in LoadInternationalStopWords.allowed_languages() else "Spanish"

    @staticmethod
    def allowed_languages():
        return [
        "English",
        "Dutch",
        "French",
        "Polish",
        "Russian",
        "Spanish",
        "Swedish"]

    def get_headers(self):
        return None

    def get_all(self):
        words = []
        for word in self.get_all_yielded():
            words.append(word)
        return words
    
    def get_all_yielded(self):
        fobj = open(self.datafile, "r", encoding="utf-8")#"utf-8-sig")
        file_reader = csv.DictReader(fobj, delimiter=',')
        for row in file_reader:
            yield row[self.lang]