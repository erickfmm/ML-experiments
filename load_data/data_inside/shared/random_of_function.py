from load_data.ILoadSupervised import ILoadSupervised
import numpy as np

__all__ = ["LoadRandom",]

def unitary_function(X):
    return 1

class LoadRandom(ILoadSupervised):
    def __init__(self, num_instances, num_dimensions, fn, min_value=0, max_value=1):
        self.num_instances = num_instances
        self.num_dimensions = num_dimensions
        self.fn = fn if fn is not None else unitary_function
        self.min_value = min_value
        self.max_value = max_value 

    def get_default(self):
        return self.get_all()

    def get_splited(self):
        return None
    
    def get_all(self):
        Xs = (np.random.random(self.num_instances * self.num_dimensions) * self.max_value) + self.min_value
        Xs = Xs.reshape(self.num_instances, self.num_dimensions)
        Ys = []
        for i in range(self.num_instances):
            Ys.append(self.fn(Xs[i]))
        return Xs, Ys
