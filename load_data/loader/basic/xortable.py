from load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadXorTable"]


class LoadXorTable(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self.xs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.ys = [1, 0, 0, 1]
        self.headers = ["first", "second"]
        self.classes = ["result"]
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
    
    def get_all(self):
        return self.xs, self.ys
    
    def get_all_yielded(self):
        for i in range(len(self.xs)):
            yield self.xs[i], self.ys[i]
