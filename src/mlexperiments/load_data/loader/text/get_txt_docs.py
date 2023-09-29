from typing import List
from mlexperiments.load_data.ILoadUnsupervised import ILoadUnsupervised
import os
import opendatasets as od

__all__ = ["LoadTXTsFolder"]


class LoadTXTsFolder(ILoadUnsupervised):
    def __init__(self, which="pei", folder_path="data/train_data/NLP_ESP/education-pei/txt_files_pei", to_catch_rbds=False):
        self.folder_path = folder_path
        self.to_catch_rbds = to_catch_rbds
        self.which = which

    def get_headers(self):
        return ["data"]

    def get_data(self) -> List[str]:
        docs = []
        self.metadata = []
        for dirname, _, filenames in os.walk(self.folder_path):
            for filename in filenames:
                #print(os.path.join(dirname, filename))
                with open(os.path.join(dirname, filename), "r", encoding="utf-8") as fh:
                    doc = fh.read()
                docs.append(doc)
                if self.to_catch_rbds:
                    rbd = int(filename.replace(".pdf.txt", "").split("_")[3])
                    self.metadata.append(rbd)
        return docs

    def download(self, folder_path="data/train_data/NLP_ESP"):
        if self.which == "pei":
            od.download("https://www.kaggle.com/datasets/erickfmm/education-pei", folder_path)
        elif self.which == "convivencia":
            od.download("https://www.kaggle.com/datasets/erickfmm/education-reglamento-convivencia", folder_path)
        elif self.which == "evaluacion":
            od.download("https://www.kaggle.com/datasets/erickfmm/education-reglamentos-de-evaluacin-txt", folder_path)
