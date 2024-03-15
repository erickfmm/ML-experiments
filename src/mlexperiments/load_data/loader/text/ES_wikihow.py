from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import json
from os.path import join
import opendatasets as od

__all__ = ["LoadWikihowSpanish"]


class LoadWikihowSpanish(ILoadSupervised):
    def __init__(self, folder_path="data/train_data/NLP_ESP/wikihow-spanish"):
        self.TYPE = SupervisedType.Classification
        self.folder_path = folder_path
    
    def get_X_Y(self):
        X = []
        Y = []
        for cuerpo, summary in self.get_X_Y_yielded():
            X.append(cuerpo)
            Y.append(summary)
        return X, Y
    
    
    def get_X_Y_yielded(self):
        with open(join(self.folder_path, "spanish.json"), "r", encoding="utf-8") as fobj:
            obj = json.load(fobj)
            for webpage in obj:
                for section in obj[webpage]:
                    yield obj[webpage][section]["document"], obj[webpage][section]["summary"]


    def get_classes(self):
        return None
    
    def get_headers(self):
        return None

    def download(self, folder_path="data/train_data/NLP_ESP"):
        od.download("https://www.kaggle.com/datasets/d410qc/wikihow-spanish", folder_path)
