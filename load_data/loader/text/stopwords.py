from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join

__all__ = ["LoadStopwords"]


class LoadStopwords(ILoadUnsupervised):
    def __init__(self, folder_path="train_data/Folder_NLPEspañol/stopword-lists-for-19-languages",
                 lang="spanish"):
        if lang not in LoadStopwords.allowed_languages():
            raise Exception("language not allowed")
        self.folder_path = folder_path
        self.filename = lang + "ST.txt"

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
        with open(join(self.folder_path, self.filename), "r") as file_obj:
            for line in file_obj:
                if len(line) > 0:
                    yield line.strip()
