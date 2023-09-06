from typing import List
from load_data.ILoadUnsupervised import ILoadUnsupervised
import os

__all__ = ["LoadTXTsFolder"]

class LoadTXTsFolder(ILoadUnsupervised):
    def __init__(self, folder_path="data/train_data/NLP_ESP/txt_files_pei", to_catch_rbds=False):
        self.folder_path = folder_path
        self.to_catch_rbds = to_catch_rbds

    def get_headers(self):
        return ["data"]

    def get_all(self) -> List[str]:
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
