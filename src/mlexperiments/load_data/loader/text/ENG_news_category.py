from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType
import json
import opendatasets as od

__all__ = ["LoadNewsCategory"]


class LoadNewsCategory(ILoadSupervised):
    def __init__(self, file_path="data/train_data/NLP_ENG/news-category-dataset/News_Category_Dataset_v3.json"):
        self.TYPE = SupervisedType.Classification
        self.file_path = file_path
        self.headers = ["headline", "short_description"]
        self.Metadata = []
        self.MetadataHeaders = ["authors", "link", "date"]

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_X_Y(self):
        file_obj = open(self.file_path, "r")
        xs = []
        ys = []
        for line in file_obj:
            line_dict = json.loads(line)
            self.Metadata.append([
                line_dict["authors"],
                line_dict["link"],
                line_dict["date"]
            ])
            y = "WORLDPOST" if line_dict["category"] == "THE WORLDPOST" else line_dict["category"]
            ys.append(y)
            xs.append([
                line_dict["headline"],
                line_dict["short_description"]
            ])
        file_obj.close()
        return xs, ys
    
    def get_classes(self):
        return None
    
    def get_headers(self):
        return self.headers

    def download(self):
        od.download("https://www.kaggle.com/datasets/rmisra/news-category-dataset", "data/train_data/NLP_ENG")
