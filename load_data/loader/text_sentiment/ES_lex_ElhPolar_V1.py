from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadElhPolarEs"]


class LoadElhPolarEs(ILoadSupervised):
    def __init__(self, file_path="train_data/Folder_NLPEspañol_Sentiment/ElhPolar_esV1.lex"):
        self.file_path = file_path
        self.classes = ["negative", "positive"]
        self.wrong = []

    def get_default(self):
        return self.get_all()

    @staticmethod
    def get_splited():
        return None
    
    def get_all(self):
        return self.read_file()
    
    def read_file(self):
        xs = []
        ys = []
        with open(self.file_path, "r") as fileobj:
            for line in fileobj:
                line2 = line.strip()
                if len(line2) == 0 or line2[0] == "#":
                    continue
                fields = line2.split("\t")
                if len(fields) == 2:
                    xs.append(fields[0].replace("_", " "))
                    ys.append(fields[1])
                else:
                    self.wrong.append(fields)
        return xs, ys

    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["phrase"]
