from load_data.ILoadUnsupervised import ILoadUnsupervised
import csv

__all__ = ["LoadInternationalStopWords"]


class LoadInternationalStopWords(ILoadUnsupervised):
    def __init__(self,
                 file_path="train_data/Folder_NLPEspañol/internationalstopwords/International-Stop-Words.csv",
                 lang="Spanish"):
        self.file_path = file_path
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
        with open(self.file_path, "r", encoding="utf-8") as file_handler: # "utf-8-sig")
            file_reader = csv.DictReader(file_handler, delimiter=',')
            for row in file_reader:
                yield row[self.lang]
