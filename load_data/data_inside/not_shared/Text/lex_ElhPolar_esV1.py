from load_data.ILoadSupervised import ILoadSupervised

__all__ = ["LoadElhPolar_es",]

class LoadElhPolar_es(ILoadSupervised):
    def __init__(self, filepath="train_data\\not_shared\\Folder_NLPEspañol_Sentiment\\ElhPolar_esV1.lex"):
        self.filepath = filepath
        self.classes = ["negative", "positive"]

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.read_file()
    
    def read_file(self):
        self.X = []
        self.Y = []
        self.wrong = []
        with open(self.filepath, "r") as fileobj:
            for line in fileobj:
                line2 = line.strip()
                if len(line2) == 0 or line2[0] == "#":
                    continue
                fields = line2.split("\t")
                if len(fields) == 2:
                    self.X.append(fields[0].replace("_", " "))
                    self.Y.append(fields[1])
                else:
                    self.wrong.append(fields)
        return self.X, self.Y

    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return ["phrase"]
