# -*- coding: utf-8 -*-

from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join

class LoadStopwords(ILoadUnsupervised):
    def __init__(self, datapath="train_data\\not_shared\\Folder_NLPEspañol\\stopword-lists-for-19-languages", \
                 lang="spanish"):
        if lang not in LoadStopwords.allowed_languages():
            raise Exception("language not allowed")
        self.datapath = datapath
        self.filename = lang+"ST.txt"

    @staticmethod
    def allowed_languages():
        return ["arabic",
                "bengali",
                "bulgarian",
                "czech",
                "english",
                "finnish",
                "french",
                "german",
                "hindi",
                "hungarian",
                "italian",
                "marathi",
                "persian",
                "polish",
                "portuguese",
                "roumanian",
                "russian",
                "spanish",
                "swedish"]
    
    def get_headers(self):
        return None

    def get_all(self):
        words = []
        for word in self.get_all_yielded():
            words.append(word)
        return words
    
    def get_all_yielded(self):
        with open(join(self.datapath, self.filename), "r") as fileobj:
            for line in fileobj:
                if len(line) > 0:
                    yield line.strip()
