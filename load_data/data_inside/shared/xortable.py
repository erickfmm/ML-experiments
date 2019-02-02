from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadXorTable",]

class LoadXorTable(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self.Xs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.Ys = [1, 0, 0, 1]
        self.headers = ["first", "second"]
        self.classes = ["result"]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers

    def get_default(self):
        return self.Xs, self.Ys

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.Xs, self.Ys
