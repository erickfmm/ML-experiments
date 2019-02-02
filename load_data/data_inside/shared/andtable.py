from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadAndTable",]

class LoadAndTable(ILoadSupervised):
        self.TYPE = SupervisedType.Classification
    def __init__(self):
        self.Xs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.Ys = [0, 0, 0, 1]
        self.headers = ["first", "second"]
        self.classes = ["result"]

    def get_default(self):
        return self.Xs, self.Ys

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.Xs, self.Ys

    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
