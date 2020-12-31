from load_data.ILoadSupervised import ILoadSupervised, SupervisedType
from numpy.random import random as np_rnd

__all__ = ["LoadRandom",]

def unitary_function(X):
    return 1

class LoadRandom(ILoadSupervised):
    def __init__(self, num_instances: int, num_dimensions: int, fn, min_value=0, max_value=1):
        self.TYPE = SupervisedType.Unknown
        self.num_instances = num_instances
        self.num_dimensions = num_dimensions
        self.fn = fn if fn is not None else unitary_function
        self.min_value = min_value
        self.max_value = max_value
        self.headers = [str(i) for i in range(num_dimensions)]
        self.classes = []
    
    def get_classes(self):
        return self.classes
    
    def get_headers(self):
        return self.headers
    
    def get_all(self):
        Xs = (np_rnd(self.num_instances * self.num_dimensions) * self.max_value) + self.min_value
        Xs = Xs.reshape(self.num_instances, self.num_dimensions)
        Ys = []
        for i in range(self.num_instances):
            Ys.append(self.fn(Xs[i]))
        return Xs, Ys
    
    def get_all_yielded(self):
        for i in range(self.num_instances):
            x = (np_rnd(self.num_dimensions) * self.max_value) + self.min_value
            y = self.fn(x[i])
            yield x, y
