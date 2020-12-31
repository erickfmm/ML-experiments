from load_data.ILoadSupervised import ILoadSupervised
import json

__all__ = ["LoadNewsCategory",]

class LoadNewsCategory(ILoadSupervised):
    def __init__(self, filepath="train_data/Folder_NLPEnglish/news-category-dataset/News_Category_Dataset_v2.json"):
        self.filepath = filepath
        self.headers = ["headline", "short_description"]

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        fileobj = open(self.filepath, "r")
        self.X = []
        self.Y = []
        self.Metadata = []
        self.MetadataHeaders = ["authors", "link", "date"]
        for line in fileobj:
            line_dict = json.loads(line)
            self.Metadata.append([
                line_dict["authors"],
                line_dict["link"],
                line_dict["date"]
            ])
            y = "WORLDPOST" if line_dict["category"] == "THE WORLDPOST" else line_dict["category"]
            self.Y.append(y)
            self.X.append([
                line_dict["headline"],
                line_dict["short_description"]
            ])
        fileobj.close()
        return self.X, self.Y
    
    def get_classes(self):
        return None #self.classes
    
    def get_headers(self):
        return self.headers
