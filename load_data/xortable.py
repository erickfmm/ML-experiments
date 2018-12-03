from load_data.ILoadSupervised import ILoadSupervised

class LoadXorTable(ILoadSupervised):
    def __init__(self):
        self.Xs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        self.Ys = [1, 0, 0, 1]

    def get_default(self):
        return self.Xs, self.Ys

    def get_splited(self):
        return None
    
    def get_all(self):
        return self.Xs, self.Ys
