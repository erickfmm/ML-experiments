class ISolution:

    def __init__(self, point, fitness):
        self.point = point
        self.fitness = fitness

    def move_to(self, new_point, new_fitness):
        self.point = new_point
        self.fitness = new_fitness
        return self.point
    
    def set_id(self, _id):
        self._id = _id
