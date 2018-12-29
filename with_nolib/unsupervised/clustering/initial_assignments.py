import random
__all__ = ["random_assignment",]

def random_assignment(self, X, num_clusters):
    assign = []
    for i in range(len(X)):
        assign.append(random.randint(0, num_clusters-1))
    return assign