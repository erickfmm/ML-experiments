from mlexperiments.load_data.ILoadSupervised import ILoadSupervised, SupervisedType

__all__ = ["LoadAndTable"]


class LoadAndTable(ILoadSupervised):
    def __init__(self):
        self.TYPE = SupervisedType.Classification
        self.xs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.ys = [0, 0, 0, 1]
    
    def get_X_Y(self):
        return self.xs, self.ys

    def get_X_Y_yielded(self):
        for i in range(len(self.xs)):
            yield self.xs[i], self.ys[i]

    def get_classes(self):
        return ["result"]
    
    def get_headers(self):
        return ["first", "second"]
