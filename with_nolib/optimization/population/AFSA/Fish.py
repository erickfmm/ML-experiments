from with_nolib.optimization.population.ISolution import ISolution

class Fish(ISolution):
    def __init__(self, _id, point, fitness):
        self.point = point
        self._id = _id
        self.fitness = fitness