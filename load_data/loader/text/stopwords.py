from load_data.ILoadUnsupervised import ILoadUnsupervised
from os.path import join

__all__ = ["LoadStopwords"]


class LoadStopwords(ILoadUnsupervised):
    def __init__(self, folder_path="data/train_data/NLP_Multilingual/stop-words-in-28-languages",
                 lang="spanish"):
        if lang not in LoadStopwords.allowed_languages():
            raise Exception("language not allowed")
        self.folder_path = folder_path
        self.filename = lang + ".txt"

    @staticmethod
    def allowed_languages():
        return [
            "arabic",
            "bulgarian",
            "catalan",
            "czech",
            "danish",
            "dutch",
            "english",
            "finnish",
            "french",
            "german",
            "gujarati",
            "hebrew",
            "hindi",
            "hungarian",
            "indonesian",
            "italian",
            "malaysian",
            "norwegian",
            "polish",
            "portuguese",
            "romanian",
            "russian",
            "slovak",
            "spanish",
            "swedish",
            "turkish",
            "ukrainian",
            "vietnamese"]
    
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

