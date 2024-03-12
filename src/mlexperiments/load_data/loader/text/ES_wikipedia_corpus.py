from mlexperiments.load_data.ILoadUnsupervised import ILoadUnsupervised
import opendatasets as od

__all__ = ["LoadES_Wikipedia_Corpus"]

class LoadES_Wikipedia_Corpus(ILoadUnsupervised):
    def __init__(self, file_path="data/train_data/NLP_ESP/corpus-de-la-wikipedia-en-espaol/eswiki-latest-pages-articles.txt"):
        self.file_path = file_path
    
    def get_data(self):
        file_obj = open(self.file_path, "r", encoding="utf-8", errors="ignore")
        content = file_obj.read()
        file_obj.close()
        return content
    
    def get_classes(self):
        return None
    
    def get_headers(self):
        return None

    def download(self, folder_path="data/train_data/NLP_ESP"):
        od.download("https://www.kaggle.com/datasets/jmorenobl/corpus-de-la-wikipedia-en-espaol", folder_path)
