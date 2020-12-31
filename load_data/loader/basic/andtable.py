from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadAndTable",]

class LoadAndTable(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self.Xs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.Ys = [0, 0, 0, 1]
        self.headers = ["first", "second"]
        self.classes = ["result"]
    
    def get_all(self):
        return self.Xs, self.Ys

    def get_all_yielded(self):
        for i in range(len(self.Xs)):
            yield self.Xs[i], self.Ys[i]

    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
