from load_data.ILoadSupervised import ILoadSupervised
from os.path import join

__all__ = ["LoadLexicons81langs"]


class LoadLexicons81langs(ILoadSupervised):

    def __init__(self, folder_path="data/train_data/NLP_ENG_Sentiment/sentiment-lexicons-for-81-languages/sentiment-lexicons/sentiment-lexicons",
                 lang="es"):
        if lang not in LoadLexicons81langs.allowed_languages():
            raise Exception("language not allowed")
        self.lang = lang
        self.folder_path = folder_path
    
    @staticmethod
    def allowed_languages():
        return ["af", "an", "ar", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy",
                "da", "de", "el", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "fy",
                "ga", "gd", "gl", "gu", "he", "hi", "hr", "ht", "hu", "hy", "ia", "id",
                "io", "is", "it", "ja", "ka", "km", "kn", "ko", "ku", "ky", "la", "lb",
                "lt", "lv", "mk", "mr", "ms", "mt", "nl", "nn", "no", "pl", "pt", "rm",
                "ro", "ru", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tk",
                "tl", "tr", "uk", "ur", "uz", "vi", "vo", "wa", "yi", "zh", "zhw"]
        
    def get_headers(self):
        return None

    def get_classes(self): 
        return ["positive", "negative"]

    def get_all(self):
        xs = []
        ys = []
        for x, y in self.get_all_yielded():
            xs.append(x)
            ys.append(y)
        return xs, ys
    
    def get_all_yielded(self):
        with open(join(self.folder_path, "negative_words_"+self.lang+".txt"), "r") as negobj:
            for line in negobj:
                if len(line) > 0:
                    yield line.strip(), 0
        # positive file
        with open(join(self.folder_path, "positive_words_"+self.lang+".txt"), "r") as posobj:
            for line in posobj:
                if len(line) > 0:
                    yield line.strip(), 1
